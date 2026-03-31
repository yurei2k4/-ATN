"""
core/models.py
--------------
Các cấu trúc dữ liệu cốt lõi của framework UTS-VRP.
Sử dụng numpy arrays (Data Flattening) để tối ưu hiệu năng cache.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


# ---------------------------------------------------------------------------
# 1. NODE – Điểm giao hàng / depot
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """
    Biểu diễn một node trong bài toán VRP.
    
    Attributes:
        id          : Chỉ số duy nhất (0 = depot)
        x, y        : Tọa độ địa lý
        demand      : Nhu cầu hàng hóa (kg hoặc đơn vị)
        ready_time  : Thời điểm sớm nhất có thể phục vụ (time window start)
        due_time    : Thời điểm trễ nhất có thể phục vụ (time window end)
        service_time: Thời gian phục vụ tại node
        node_type   : 'depot' hoặc 'customer'
    """
    id: int
    x: float
    y: float
    demand: float = 0.0
    ready_time: float = 0.0
    due_time: float = float('inf')
    service_time: float = 0.0
    node_type: str = 'customer'

    def __repr__(self):
        return f"Node(id={self.id}, demand={self.demand}, tw=[{self.ready_time},{self.due_time}])"


# ---------------------------------------------------------------------------
# 2. VEHICLE – Xe vận chuyển
# ---------------------------------------------------------------------------

@dataclass
class Vehicle:
    """
    Biểu diễn một xe trong đội xe.
    
    Attributes:
        id          : Chỉ số xe
        capacity    : Tải trọng tối đa
        depot_id    : Depot xuất phát (hỗ trợ Multi-depot)
        vehicle_type: Loại xe (ảnh hưởng đến phí cầu đường, tốc độ)
        max_duration: Thời gian tối đa một chuyến (tùy chọn)
    """
    id: int
    capacity: float
    depot_id: int = 0
    vehicle_type: str = 'standard'
    max_duration: float = float('inf')

    def __repr__(self):
        return f"Vehicle(id={self.id}, cap={self.capacity}, depot={self.depot_id})"


# ---------------------------------------------------------------------------
# 3. PROBLEM – Bài toán VRP đầu vào
# ---------------------------------------------------------------------------

@dataclass
class VRPProblem:
    """
    Đóng gói toàn bộ dữ liệu đầu vào của bài toán VRP.
    
    Attributes:
        nodes           : Danh sách tất cả nodes (index 0 = depot)
        vehicles        : Danh sách xe
        dist_matrix     : Ma trận khoảng cách [n x n] numpy array
                          Có thể bất đối xứng (Asymmetric)
        time_matrix     : Ma trận thời gian di chuyển (tùy chọn)
        problem_type    : 'CVRP' | 'VRPTW' | 'AVRP' | 'MDVRP'
        name            : Tên bài toán (ví dụ: 'C101')
    """
    nodes: List[Node]
    vehicles: List[Vehicle]
    dist_matrix: np.ndarray
    time_matrix: Optional[np.ndarray] = None
    problem_type: str = 'CVRP'
    name: str = 'unnamed'

    def __post_init__(self):
        # Nếu không có time_matrix, dùng dist_matrix (giả sử tốc độ = 1)
        if self.time_matrix is None:
            self.time_matrix = self.dist_matrix.copy()
        self._validate()

    def _validate(self):
        n = len(self.nodes)
        assert self.dist_matrix.shape == (n, n), \
            f"dist_matrix phải có shape ({n}, {n}), nhận được {self.dist_matrix.shape}"

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_vehicles(self) -> int:
        return len(self.vehicles)

    @property
    def depot(self) -> Node:
        return self.nodes[0]

    @property
    def customers(self) -> List[Node]:
        return [n for n in self.nodes if n.node_type == 'customer']

    def get_dist(self, i: int, j: int) -> float:
        """Lấy khoảng cách từ node i đến node j (hỗ trợ bất đối xứng)."""
        return float(self.dist_matrix[i, j])

    def get_time(self, i: int, j: int) -> float:
        """Lấy thời gian di chuyển từ node i đến node j."""
        return float(self.time_matrix[i, j])

    def euclidean_dist_matrix(self) -> np.ndarray:
        """Tính lại ma trận Euclidean từ tọa độ (dùng khi không có ma trận thực tế)."""
        coords = np.array([[n.x, n.y] for n in self.nodes])
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))


# ---------------------------------------------------------------------------
# 4. ROUTE – Một lộ trình của một xe
# ---------------------------------------------------------------------------

class Route:
    """
    Biểu diễn lộ trình của một xe dưới dạng mảng phẳng (flat array)
    để tối ưu tốc độ truy cập (cache-friendly).
    
    Lộ trình luôn bắt đầu và kết thúc tại depot.
    Ví dụ: [0, 3, 7, 2, 0] → depot → node3 → node7 → node2 → depot
    """

    def __init__(self, vehicle: Vehicle, depot_id: int = 0):
        self.vehicle = vehicle
        self.depot_id = depot_id
        # Mảng lưu trữ chính – luôn bắt đầu và kết thúc bằng depot
        self._nodes: List[int] = [depot_id, depot_id]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> List[int]:
        """Trả về danh sách node IDs trong lộ trình (kể cả depot đầu/cuối)."""
        return self._nodes

    @property
    def customers(self) -> List[int]:
        """Chỉ các customer nodes (không kể depot)."""
        return self._nodes[1:-1]

    @property
    def num_customers(self) -> int:
        return len(self._nodes) - 2

    @property
    def is_empty(self) -> bool:
        return self.num_customers == 0

    # ------------------------------------------------------------------
    # Modification
    # ------------------------------------------------------------------

    def insert(self, position: int, node_id: int):
        """Chèn node_id vào vị trí position (1-indexed trong phần customers)."""
        self._nodes.insert(position, node_id)

    def remove(self, position: int) -> int:
        """Xóa và trả về node tại position."""
        return self._nodes.pop(position)

    def append_customer(self, node_id: int):
        """Thêm customer vào cuối lộ trình (trước depot cuối)."""
        self._nodes.insert(len(self._nodes) - 1, node_id)

    # ------------------------------------------------------------------
    # Metrics (cần VRPProblem để tính)
    # ------------------------------------------------------------------

    def total_distance(self, problem: VRPProblem) -> float:
        """Tổng khoảng cách lộ trình."""
        total = 0.0
        for i in range(len(self._nodes) - 1):
            total += problem.get_dist(self._nodes[i], self._nodes[i + 1])
        return total

    def total_load(self, problem: VRPProblem) -> float:
        """Tổng tải trọng lộ trình."""
        return sum(problem.nodes[nid].demand for nid in self.customers)

    def arrival_times(self, problem: VRPProblem) -> List[float]:
        """
        Tính thời điểm đến tại mỗi node trong lộ trình.
        Có xét time window: xe chờ nếu đến sớm hơn ready_time.
        """
        times = [0.0]  # depot xuất phát lúc 0
        current_time = 0.0
        for i in range(len(self._nodes) - 1):
            src = self._nodes[i]
            dst = self._nodes[i + 1]
            current_time += problem.get_time(src, dst)
            node = problem.nodes[dst]
            # Nếu đến sớm hơn ready_time → chờ
            current_time = max(current_time, node.ready_time)
            current_time += node.service_time
            times.append(current_time)
        return times

    def copy(self) -> 'Route':
        """Deep copy lộ trình."""
        new_route = Route(self.vehicle, self.depot_id)
        new_route._nodes = self._nodes.copy()
        return new_route

    def __repr__(self):
        return f"Route(vehicle={self.vehicle.id}, nodes={self._nodes}, customers={self.num_customers})"

    def __len__(self):
        return self.num_customers


# ---------------------------------------------------------------------------
# 5. SOLUTION – Tập hợp các lộ trình
# ---------------------------------------------------------------------------

class Solution:
    """
    Biểu diễn một giải pháp VRP hoàn chỉnh: tập hợp các Route.
    
    Duy trì mapping customer → route_index để O(1) lookup.
    """

    def __init__(self, problem: VRPProblem):
        self.problem = problem
        self.routes: List[Route] = []
        # customer_id → (route_index, position_in_route)
        self._customer_map: Dict[int, tuple] = {}

    # ------------------------------------------------------------------
    # Xây dựng solution
    # ------------------------------------------------------------------

    def add_route(self, route: Route):
        """Thêm một lộ trình vào solution."""
        route_idx = len(self.routes)
        self.routes.append(route)
        self._rebuild_customer_map_for_route(route_idx)

    def _rebuild_customer_map(self):
        """Rebuild toàn bộ customer map."""
        self._customer_map = {}
        for r_idx, route in enumerate(self.routes):
            self._rebuild_customer_map_for_route(r_idx)

    def _rebuild_customer_map_for_route(self, route_idx: int):
        """Rebuild customer map cho một route."""
        route = self.routes[route_idx]
        for pos, node_id in enumerate(route.nodes):
            if node_id != route.depot_id:
                self._customer_map[node_id] = (route_idx, pos)

    # ------------------------------------------------------------------
    # Objective Function
    # ------------------------------------------------------------------

    def total_distance(self) -> float:
        """Tổng khoảng cách tất cả các lộ trình."""
        return sum(r.total_distance(self.problem) for r in self.routes)

    def num_vehicles_used(self) -> int:
        """Số xe được sử dụng (lộ trình không rỗng)."""
        return sum(1 for r in self.routes if not r.is_empty)

    def augmented_objective(self, lambdas: Dict[str, float], violations: Dict[str, float]) -> float:
        """
        Hàm mục tiêu mở rộng:
        F(s) = C(s) + Σ λ_i * V_i(s)
        
        Args:
            lambdas    : {tên_ràng_buộc: hệ_số_phạt}
            violations : {tên_ràng_buộc: giá_trị_vi_phạm}
        """
        penalty = sum(lambdas.get(k, 0.0) * v for k, v in violations.items())
        return self.total_distance() + penalty

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_route_of(self, customer_id: int) -> tuple:
        """Trả về (route_index, position) của customer."""
        return self._customer_map.get(customer_id, (-1, -1))

    def is_feasible(self, plugins: List) -> bool:
        """Kiểm tra tính khả thi theo tất cả plugins."""
        for plugin in plugins:
            total_viol = sum(plugin.compute_violation(r, self.problem) for r in self.routes)
            if total_viol > 1e-6:
                return False
        return True

    def copy(self) -> 'Solution':
        """Deep copy solution."""
        new_sol = Solution(self.problem)
        new_sol.routes = [r.copy() for r in self.routes]
        new_sol._rebuild_customer_map()
        return new_sol

    def to_dict(self) -> Dict[str, Any]:
        """Xuất solution dưới dạng dict (để lưu/visualize)."""
        return {
            'total_distance': self.total_distance(),
            'num_vehicles': self.num_vehicles_used(),
            'routes': [
                {
                    'vehicle_id': r.vehicle.id,
                    'nodes': r.nodes,
                    'customers': r.customers,
                    'distance': r.total_distance(self.problem),
                    'load': r.total_load(self.problem),
                }
                for r in self.routes if not r.is_empty
            ]
        }

    def __repr__(self):
        dist = self.total_distance()
        return f"Solution(routes={self.num_vehicles_used()}, total_dist={dist:.2f})"