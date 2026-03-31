"""
plugins/capacity.py
-------------------
CapacityPlugin: Ràng buộc tải trọng xe (CVRP).

Vi phạm = max(0, total_load - capacity)

Delta Evaluation:
    Khi relocate node u từ route A sang route B:
    - violation(A) thay đổi: load_A giảm demand[u]
    - violation(B) thay đổi: load_B tăng demand[u]
    → Chỉ cần cộng/trừ demand[u], không cần duyệt lại toàn bộ route
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any
from plugins.base import IConstraintPlugin

if TYPE_CHECKING:
    from core.models import Route, VRPProblem, Solution


class CapacityPlugin(IConstraintPlugin):
    """
    Ràng buộc tải trọng cho CVRP.
    
    Ràng buộc CỨNG (Hard Constraint):
        Σ demand[i] ≤ vehicle.capacity  với mọi xe
    
    Sử dụng Segment-based Delta Evaluation để tính nhanh.
    """

    def __init__(self, violation_scale: float = 1.0):
        """
        Args:
            violation_scale: Hệ số scale violation
                             (để normalize so với các constraints khác)
        """
        self._violation_scale = violation_scale
        # Cache: route_id → current_load
        self._load_cache: Dict[int, float] = {}

    @property
    def name(self) -> str:
        return 'capacity'

    @property
    def priority(self) -> int:
        return 10  # Ưu tiên cao – tính trước để short-circuit sớm

    def compute_violation(self, route: 'Route', problem: 'VRPProblem') -> float:
        """
        V_capacity = max(0, total_load - capacity) * scale
        """
        total_load = sum(
            problem.nodes[nid].demand
            for nid in route.customers
        )
        capacity = route.vehicle.capacity
        excess = max(0.0, total_load - capacity)
        return excess * self._violation_scale

    def delta_violation(
        self,
        route: 'Route',
        problem: 'VRPProblem',
        move_info: Dict[str, Any],
        current_violation: float
    ) -> float:
        """
        Delta Evaluation cho capacity:
        Nếu move_info có 'load_delta' → tính trong O(1).
        
        move_info keys:
            - 'load_delta': thay đổi tải trọng (+ thêm, - bớt)
            - 'current_load': tải trọng hiện tại của route (tùy chọn)
        """
        if 'load_delta' in move_info and 'current_load' in move_info:
            new_load = move_info['current_load'] + move_info['load_delta']
            capacity = route.vehicle.capacity
            excess = max(0.0, new_load - capacity)
            return excess * self._violation_scale

        # Fallback: tính lại toàn bộ
        return self.compute_violation(route, problem)

    def on_solution_start(self, solution: 'Solution'):
        """Cache load của tất cả routes."""
        self._load_cache = {
            i: route.total_load(solution.problem)
            for i, route in enumerate(solution.routes)
        }

    def on_move_accepted(self, move_info: Dict[str, Any]):
        """Cập nhật cache sau khi move được chấp nhận."""
        if 'route_idx' in move_info and 'load_delta' in move_info:
            r_idx = move_info['route_idx']
            if r_idx in self._load_cache:
                self._load_cache[r_idx] += move_info['load_delta']

    def get_route_load(self, route_idx: int) -> float:
        """Lấy load từ cache (nếu có)."""
        return self._load_cache.get(route_idx, 0.0)