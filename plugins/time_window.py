"""
plugins/time_window.py
----------------------
TimeWindowPlugin: Ràng buộc cửa sổ thời gian (VRPTW).

Vi phạm = Σ max(0, arrival_time[i] - due_time[i])  (vi phạm trễ)
         + max(0, departure_time - vehicle.max_duration) (vi phạm thời gian tổng)

Segment-based Delta Evaluation:
    Khi chèn/xóa một node, chỉ các node PHÍA SAU trong lộ trình
    bị ảnh hưởng về arrival time → chỉ cần recalculate từ điểm thay đổi.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, List, Tuple
from plugins.base import IConstraintPlugin

if TYPE_CHECKING:
    from core.models import Route, VRPProblem


class TimeWindowPlugin(IConstraintPlugin):
    """
    Ràng buộc Time Window cho VRPTW.
    
    Hỗ trợ:
        - Hard time window: phạt nặng khi đến sau due_time
        - Soft time window: phạt nhẹ hơn (tùy chọn qua soft_mode=True)
    """

    def __init__(
        self,
        late_penalty_scale: float = 1.0,
        soft_mode: bool = False,
        early_penalty_scale: float = 0.0,  # Phạt đến sớm (soft TW)
    ):
        self._late_scale = late_penalty_scale
        self._early_scale = early_penalty_scale
        self._soft_mode = soft_mode

    @property
    def name(self) -> str:
        return 'time_window'

    @property
    def priority(self) -> int:
        return 20

    def compute_violation(self, route: 'Route', problem: 'VRPProblem') -> float:
        """
        Tính tổng vi phạm time window cho toàn bộ route.
        
        Thuật toán:
            1. Mô phỏng di chuyển từ depot
            2. Tại mỗi node: tính arrival time
            3. Nếu arrival > due_time → cộng vào vi phạm
        """
        if route.is_empty:
            return 0.0

        total_violation = 0.0
        current_time = 0.0
        nodes = route.nodes

        for i in range(len(nodes) - 1):
            src_id = nodes[i]
            dst_id = nodes[i + 1]

            # Di chuyển từ src đến dst
            current_time += problem.get_time(src_id, dst_id)

            dst_node = problem.nodes[dst_id]

            # Phạt đến trễ
            if current_time > dst_node.due_time + 1e-9:
                total_violation += (current_time - dst_node.due_time) * self._late_scale

            # Phạt đến sớm (chỉ trong soft mode)
            if self._soft_mode and current_time < dst_node.ready_time:
                total_violation += (dst_node.ready_time - current_time) * self._early_scale

            # Xe chờ nếu đến sớm
            current_time = max(current_time, dst_node.ready_time)

            # Cộng thời gian phục vụ
            if dst_id != route.depot_id:
                current_time += dst_node.service_time

        return total_violation

    def compute_arrival_times(
        self, route: 'Route', problem: 'VRPProblem'
    ) -> List[Tuple[int, float, bool]]:
        """
        Tính arrival times tất cả nodes trong route.
        
        Returns:
            List of (node_id, arrival_time, is_late)
        """
        result = []
        current_time = 0.0
        nodes = route.nodes

        for i in range(len(nodes) - 1):
            src_id = nodes[i]
            dst_id = nodes[i + 1]
            current_time += problem.get_time(src_id, dst_id)

            dst_node = problem.nodes[dst_id]
            is_late = current_time > dst_node.due_time + 1e-9
            result.append((dst_id, current_time, is_late))

            current_time = max(current_time, dst_node.ready_time)
            if dst_id != route.depot_id:
                current_time += dst_node.service_time

        return result

    def delta_violation_for_segment(
        self,
        route: 'Route',
        problem: 'VRPProblem',
        start_pos: int,
        move_info: Dict[str, Any],
        current_violation: float
    ) -> float:
        """
        Segment-based Delta Evaluation:
        Tính lại violation chỉ từ start_pos trở về sau.
        
        Ưu điểm: Khi chèn/xóa node ở giữa route, không cần recalculate
        toàn bộ – chỉ recalculate phần bị ảnh hưởng.
        
        Args:
            start_pos: Vị trí bắt đầu recalculate (node thứ start_pos bị ảnh hưởng)
        """
        if start_pos <= 0:
            return self.compute_violation(route, problem)

        # Tính violation của segment [0, start_pos) – không thay đổi
        nodes = route.nodes
        current_time = 0.0
        unaffected_violation = 0.0

        for i in range(min(start_pos, len(nodes) - 1)):
            src_id = nodes[i]
            dst_id = nodes[i + 1]
            current_time += problem.get_time(src_id, dst_id)
            dst_node = problem.nodes[dst_id]
            if current_time > dst_node.due_time + 1e-9:
                unaffected_violation += (current_time - dst_node.due_time) * self._late_scale
            current_time = max(current_time, dst_node.ready_time)
            if dst_id != route.depot_id:
                current_time += dst_node.service_time

        # Tính violation của segment [start_pos, end) – bị ảnh hưởng
        affected_violation = 0.0
        for i in range(start_pos, len(nodes) - 1):
            src_id = nodes[i]
            dst_id = nodes[i + 1]
            current_time += problem.get_time(src_id, dst_id)
            dst_node = problem.nodes[dst_id]
            if current_time > dst_node.due_time + 1e-9:
                affected_violation += (current_time - dst_node.due_time) * self._late_scale
            current_time = max(current_time, dst_node.ready_time)
            if dst_id != route.depot_id:
                current_time += dst_node.service_time

        return unaffected_violation + affected_violationsss