"""
plugins/asymmetric.py
---------------------
AsymmetricRoutePlugin: Xử lý đường một chiều và ma trận khoảng cách bất đối xứng.

Đặc thù Logistics Việt Nam:
    - Đường một chiều (one-way streets)
    - Khoảng cách từ A→B ≠ B→A
    - Một số tuyến đường có phí (cầu đường, hầm vượt sông)
    - Giờ cấm tải (trucks not allowed during peak hours)

Ma trận bất đối xứng: dist[i][j] ≠ dist[j][i]
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, Optional, List, Tuple
from plugins.base import IConstraintPlugin
import numpy as np

if TYPE_CHECKING:
    from core.models import Route, VRPProblem


class AsymmetricRoutePlugin(IConstraintPlugin):
    """
    Plugin xử lý ràng buộc đường bất đối xứng.
    
    Tính năng:
        1. Validate rằng solver dùng asymmetric dist_matrix
        2. Thêm phí cầu đường / tuyến đường có phí
        3. Kiểm tra ràng buộc giờ cấm tải
    """

    def __init__(
        self,
        toll_matrix: Optional[np.ndarray] = None,
        restricted_hours: Optional[List[Tuple[float, float]]] = None,
        violation_scale: float = 1.0,
    ):
        """
        Args:
            toll_matrix     : Ma trận phí cầu đường [n x n]
                              toll_matrix[i][j] = phí khi đi từ i → j
            restricted_hours: Danh sách khoảng giờ cấm tải
                              [(start_hour, end_hour), ...]
                              Ví dụ: [(6.5, 8.5), (17.0, 19.0)] = sáng và chiều
            violation_scale : Hệ số scale vi phạm
        """
        self._toll_matrix = toll_matrix
        self._restricted_hours = restricted_hours or []
        self._violation_scale = violation_scale

    @property
    def name(self) -> str:
        return 'asymmetric'

    @property
    def priority(self) -> int:
        return 30

    def compute_violation(self, route: 'Route', problem: 'VRPProblem') -> float:
        """
        Tính vi phạm giờ cấm tải:
        Mỗi node được phục vụ trong giờ cấm = vi phạm
        """
        if not self._restricted_hours or route.is_empty:
            return 0.0

        total_violation = 0.0
        current_time = 0.0
        nodes = route.nodes

        for i in range(len(nodes) - 1):
            src_id = nodes[i]
            dst_id = nodes[i + 1]
            current_time += problem.get_time(src_id, dst_id)

            dst_node = problem.nodes[dst_id]

            # Kiểm tra có đang trong giờ cấm không
            time_in_hours = current_time / 60.0  # Giả sử time_matrix tính bằng phút
            for start_ban, end_ban in self._restricted_hours:
                if start_ban <= time_in_hours <= end_ban:
                    total_violation += self._violation_scale
                    break

            current_time = max(current_time, dst_node.ready_time)
            current_time += dst_node.service_time

        return total_violation

    def compute_cost(self, route: 'Route', problem: 'VRPProblem') -> float:
        """
        Tính chi phí phụ: phí cầu đường theo tuyến đường.
        """
        if self._toll_matrix is None or route.is_empty:
            return 0.0

        total_toll = 0.0
        nodes = route.nodes
        for i in range(len(nodes) - 1):
            src_id = nodes[i]
            dst_id = nodes[i + 1]
            total_toll += float(self._toll_matrix[src_id, dst_id])
        return total_toll

    def verify_asymmetric(self, problem: 'VRPProblem') -> bool:
        """
        Kiểm tra dist_matrix có thực sự bất đối xứng không.
        (Để debug / validation)
        """
        dist = problem.dist_matrix
        diff = np.abs(dist - dist.T)
        return bool(np.any(diff > 1e-6))

    def compute_route_distance_asymmetric(
        self, route: 'Route', problem: 'VRPProblem'
    ) -> float:
        """
        Tính khoảng cách lộ trình với ma trận bất đối xứng.
        Kết quả sẽ khác so với phiên bản đối xứng.
        """
        total = 0.0
        nodes = route.nodes
        for i in range(len(nodes) - 1):
            total += problem.get_dist(nodes[i], nodes[i + 1])
        return total