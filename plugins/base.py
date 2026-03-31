"""
plugins/base.py
---------------
Interface IConstraintPlugin theo nguyên lý IoC (Inversion of Control).

Mọi constraint đều implement interface này.
Core Solver không biết gì về logic của từng constraint –
nó chỉ gọi các phương thức chuẩn hóa.

Cơ chế Delta Evaluation:
    Thay vì tính lại toàn bộ lộ trình sau mỗi move,
    chỉ tính phần thay đổi (delta) dựa trên các segments bị ảnh hưởng.
    
    Ví dụ: Sau move Relocate(node=5, from_route=0, to_route=1):
    → Chỉ cần tính lại segment [4, 5, 6] trong route 0
      và segment [prev, 5, next] trong route 1
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Optional

if TYPE_CHECKING:
    from core.models import Route, VRPProblem, Solution


class IConstraintPlugin(ABC):
    """
    Interface chuẩn cho tất cả constraint plugins.
    
    Mỗi plugin phải implement:
        - name          : Tên định danh (ví dụ: 'capacity')
        - compute_violation(route, problem) → float
        - compute_cost(route, problem) → float  [tùy chọn]
        - delta_violation(route, problem, move_info) → float  [tối ưu hóa]
        - on_solution_start(solution)  [khởi tạo state cache]
        - on_move_accepted(move_info)  [cập nhật state cache]
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Tên định danh duy nhất của plugin."""
        ...

    @property
    def priority(self) -> int:
        """
        Thứ tự ưu tiên khi tính toán (thấp hơn = tính trước).
        Dùng để tối ưu short-circuit evaluation.
        """
        return 100

    # ------------------------------------------------------------------
    # Core Methods
    # ------------------------------------------------------------------

    @abstractmethod
    def compute_violation(self, route: 'Route', problem: 'VRPProblem') -> float:
        """
        Tính mức độ vi phạm của route.
        
        Returns:
            0.0 nếu không vi phạm
            > 0 nếu vi phạm (giá trị càng cao = vi phạm càng nặng)
        """
        ...

    def compute_cost(self, route: 'Route', problem: 'VRPProblem') -> float:
        """
        Tính chi phí phụ của route (ví dụ: phí cầu đường).
        Mặc định = 0 (không có chi phí phụ).
        """
        return 0.0

    # ------------------------------------------------------------------
    # Delta Evaluation (Key Performance Feature)
    # ------------------------------------------------------------------

    def delta_violation(
        self,
        route: 'Route',
        problem: 'VRPProblem',
        move_info: Dict[str, Any],
        current_violation: float
    ) -> float:
        """
        Tính NHANH sự thay đổi violation sau một move.
        
        Mặc định: tính lại toàn bộ (fallback).
        Các plugin nên override với implementation tối ưu hơn.
        
        Args:
            route           : Route sau khi áp dụng move
            problem         : VRP Problem
            move_info       : Thông tin move (segments bị ảnh hưởng)
            current_violation: Violation hiện tại trước move
        
        Returns:
            Violation mới sau move
        """
        return self.compute_violation(route, problem)

    # ------------------------------------------------------------------
    # State Management (cho caching)
    # ------------------------------------------------------------------

    def on_solution_start(self, solution: 'Solution'):
        """
        Hook gọi khi bắt đầu một solution mới.
        Dùng để khởi tạo cache nội bộ.
        """
        pass

    def on_move_accepted(self, move_info: Dict[str, Any]):
        """
        Hook gọi khi một move được chấp nhận.
        Dùng để cập nhật incremental state.
        """
        pass

    def on_solution_copied(self, source_solution: 'Solution', target_solution: 'Solution'):
        """Hook khi solution được copy (deep copy)."""
        pass

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Trả về cấu hình hiện tại của plugin."""
        return {'name': self.name, 'priority': self.priority}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class PluginRegistry:
    """
    Registry quản lý tất cả plugins theo nguyên lý IoC.
    Core Solver inject plugins thông qua registry này.
    """

    def __init__(self):
        self._plugins: Dict[str, IConstraintPlugin] = {}

    def register(self, plugin: IConstraintPlugin):
        """Đăng ký một plugin."""
        self._plugins[plugin.name] = plugin
        return self  # fluent interface

    def unregister(self, name: str):
        """Hủy đăng ký plugin."""
        self._plugins.pop(name, None)

    def get(self, name: str) -> Optional[IConstraintPlugin]:
        """Lấy plugin theo tên."""
        return self._plugins.get(name)

    def all(self) -> list:
        """Lấy tất cả plugins, sắp xếp theo priority."""
        return sorted(self._plugins.values(), key=lambda p: p.priority)

    def compute_all_violations(self, route: 'Route', problem: 'VRPProblem') -> Dict[str, float]:
        """Tính violations từ tất cả plugins cho một route."""
        return {
            plugin.name: plugin.compute_violation(route, problem)
            for plugin in self.all()
        }

    def total_violation(self, route: 'Route', problem: 'VRPProblem') -> float:
        """Tổng violations (unweighted)."""
        return sum(self.compute_all_violations(route, problem).values())

    def is_feasible(self, route: 'Route', problem: 'VRPProblem') -> bool:
        """Kiểm tra route có feasible không."""
        return all(
            plugin.compute_violation(route, problem) < 1e-9
            for plugin in self.all()
        )

    def __len__(self):
        return len(self._plugins)

    def __repr__(self):
        names = list(self._plugins.keys())
        return f"PluginRegistry(plugins={names})"