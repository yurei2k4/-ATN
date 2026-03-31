"""
operators/inter_route.py
------------------------
Toán tử lân cận giữa các lộ trình (Inter-route operators).

Đây là nhóm operator quan trọng nhất để tối ưu số xe sử dụng.

Nhóm Inter-route:
    - Relocate    : Di chuyển 1 node từ route này sang route khác
    - Swap        : Hoán đổi 2 nodes giữa 2 routes
    - CrossExchange: Trao đổi 2 đoạn (segments) giữa 2 routes
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Dict, Any
import numpy as np

from operators.intra_route import BaseOperator, Move

if TYPE_CHECKING:
    from core.models import Solution, VRPProblem, Route


# ===========================================================================
# RELOCATE OPERATOR
# ===========================================================================

@dataclass
class RelocateMove(Move):
    """
    Relocate: Di chuyển node u từ route_from sang route_to.
    
    Trước: route_from = [..., prev_u, u, next_u, ...]
           route_to   = [..., prev_v, next_v, ...]
    
    Sau:   route_from = [..., prev_u, next_u, ...]
           route_to   = [..., prev_v, u, next_v, ...]
    """
    node_id: int = 0
    route_from: int = 0
    route_to: int = 0
    pos_from: int = 0       # Vị trí của u trong route_from
    pos_to: int = 0         # Vị trí chèn vào trong route_to
    demand: float = 0.0     # Demand của node (cho Delta capacity)


class RelocateOperator(BaseOperator):
    """
    Relocate operator: move một node giữa 2 routes.
    
    Đây là operator mạnh nhất để giảm số xe (vehicle minimization).
    Khi route_from chỉ còn 1 node sau relocate → route đó được giải phóng.
    """

    def __init__(self, use_candidate_list: bool = True, candidate_size: int = 10):
        """
        Args:
            use_candidate_list: Chỉ xét các routes "gần" nhất cho to_route
                                để giảm không gian tìm kiếm
            candidate_size    : Số routes candidate xét cho mỗi node
        """
        self.use_candidate_list = use_candidate_list
        self.candidate_size = candidate_size

    def generate_moves(
        self,
        solution: 'Solution',
        problem: 'VRPProblem',
        max_moves: Optional[int] = None
    ) -> List[RelocateMove]:
        moves = []
        dist = problem.dist_matrix
        routes = solution.routes
        n_routes = len(routes)

        for r_from in range(n_routes):
            route_from = routes[r_from]
            if route_from.is_empty:
                continue

            nodes_from = route_from.nodes
            # Duyệt từng customer trong route_from
            for pos_from in range(1, len(nodes_from) - 1):
                u = nodes_from[pos_from]
                prev_u = nodes_from[pos_from - 1]
                next_u = nodes_from[pos_from + 1]

                # Chi phí xóa u khỏi route_from
                removal_delta = (
                    -dist[prev_u, u]
                    -dist[u, next_u]
                    +dist[prev_u, next_u]
                )

                demand_u = problem.nodes[u].demand

                # Thử chèn u vào các route khác
                for r_to in range(n_routes):
                    if r_to == r_from:
                        continue

                    route_to = routes[r_to]
                    nodes_to = route_to.nodes

                    # Kiểm tra sơ bộ capacity (short-circuit pruning)
                    cap_to = route_to.vehicle.capacity
                    load_to = route_to.total_load(problem)
                    if load_to + demand_u > cap_to + 1e-6:
                        continue  # Chắc chắn vi phạm capacity → skip

                    # Thử tất cả vị trí chèn trong route_to
                    for pos_to in range(1, len(nodes_to)):
                        prev_v = nodes_to[pos_to - 1]
                        next_v = nodes_to[pos_to]

                        insertion_delta = (
                            -dist[prev_v, next_v]
                            +dist[prev_v, u]
                            +dist[u, next_v]
                        )

                        delta = removal_delta + insertion_delta

                        move = RelocateMove(
                            move_type='relocate',
                            delta_cost=delta,
                            move_hash=hash((u, r_from, r_to, pos_to, 'relocate')),
                            node_id=u,
                            route_from=r_from,
                            route_to=r_to,
                            pos_from=pos_from,
                            pos_to=pos_to,
                            demand=demand_u,
                            metadata={
                                'load_delta_from': -demand_u,
                                'load_delta_to': demand_u,
                                'current_load_from': load_to,
                                'current_load_to': load_to,
                            }
                        )
                        moves.append(move)

                        if max_moves and len(moves) >= max_moves:
                            return moves

        return moves

    def apply(self, solution: 'Solution', move: RelocateMove) -> 'Solution':
        """
        Áp dụng Relocate:
        1. Xóa node khỏi route_from tại pos_from
        2. Chèn node vào route_to tại pos_to
        """
        new_sol = solution.copy()

        # Bước 1: Xóa khỏi route_from
        route_from = new_sol.routes[move.route_from]
        route_from.nodes.pop(move.pos_from)

        # Bước 2: Chèn vào route_to
        # Điều chỉnh pos_to nếu cùng route (trường hợp intra, nhưng ở đây inter)
        route_to = new_sol.routes[move.route_to]
        pos_to = min(move.pos_to, len(route_to.nodes))
        route_to.nodes.insert(pos_to, move.node_id)

        new_sol._rebuild_customer_map()
        return new_sol

    def undo(self, solution: 'Solution', move: RelocateMove) -> 'Solution':
        """Undo Relocate: tạo reverse move."""
        reverse = RelocateMove(
            move_type='relocate',
            delta_cost=-move.delta_cost,
            node_id=move.node_id,
            route_from=move.route_to,
            route_to=move.route_from,
            pos_from=move.pos_to,
            pos_to=move.pos_from,
            demand=move.demand,
        )
        return self.apply(solution, reverse)


# ===========================================================================
# SWAP OPERATOR
# ===========================================================================

@dataclass
class SwapMove(Move):
    """
    Swap: Hoán đổi node u (trong route_a) và node v (trong route_b).
    
    Trước: route_a = [..., prev_u, u, next_u, ...]
           route_b = [..., prev_v, v, next_v, ...]
    
    Sau:   route_a = [..., prev_u, v, next_u, ...]
           route_b = [..., prev_v, u, next_v, ...]
    """
    node_u: int = 0
    node_v: int = 0
    route_a: int = 0
    route_b: int = 0
    pos_u: int = 0
    pos_v: int = 0
    demand_u: float = 0.0
    demand_v: float = 0.0


class SwapOperator(BaseOperator):
    """
    Swap operator: hoán đổi 2 nodes giữa 2 routes.
    Hữu ích khi cần cân bằng tải trọng giữa các xe.
    """

    def generate_moves(
        self,
        solution: 'Solution',
        problem: 'VRPProblem',
        max_moves: Optional[int] = None
    ) -> List[SwapMove]:
        moves = []
        dist = problem.dist_matrix
        routes = solution.routes
        n_routes = len(routes)

        for r_a in range(n_routes):
            route_a = routes[r_a]
            if route_a.is_empty:
                continue
            nodes_a = route_a.nodes

            for r_b in range(r_a + 1, n_routes):
                route_b = routes[r_b]
                if route_b.is_empty:
                    continue
                nodes_b = route_b.nodes

                for pos_u in range(1, len(nodes_a) - 1):
                    u = nodes_a[pos_u]
                    prev_u = nodes_a[pos_u - 1]
                    next_u = nodes_a[pos_u + 1]
                    demand_u = problem.nodes[u].demand

                    for pos_v in range(1, len(nodes_b) - 1):
                        v = nodes_b[pos_v]
                        prev_v = nodes_b[pos_v - 1]
                        next_v = nodes_b[pos_v + 1]
                        demand_v = problem.nodes[v].demand

                        # Kiểm tra capacity sơ bộ
                        load_a = route_a.total_load(problem)
                        load_b = route_b.total_load(problem)
                        if (load_a - demand_u + demand_v > route_a.vehicle.capacity + 1e-6 or
                                load_b - demand_v + demand_u > route_b.vehicle.capacity + 1e-6):
                            continue  # Short-circuit

                        # Tính delta khoảng cách
                        old_cost = (dist[prev_u, u] + dist[u, next_u] +
                                    dist[prev_v, v] + dist[v, next_v])
                        new_cost = (dist[prev_u, v] + dist[v, next_u] +
                                    dist[prev_v, u] + dist[u, next_v])
                        delta = new_cost - old_cost

                        move = SwapMove(
                            move_type='swap',
                            delta_cost=delta,
                            move_hash=hash((u, v, r_a, r_b, 'swap')),
                            node_u=u,
                            node_v=v,
                            route_a=r_a,
                            route_b=r_b,
                            pos_u=pos_u,
                            pos_v=pos_v,
                            demand_u=demand_u,
                            demand_v=demand_v,
                        )
                        moves.append(move)

                        if max_moves and len(moves) >= max_moves:
                            return moves

        return moves

    def apply(self, solution: 'Solution', move: SwapMove) -> 'Solution':
        """Hoán đổi u và v giữa 2 routes."""
        new_sol = solution.copy()
        nodes_a = new_sol.routes[move.route_a].nodes
        nodes_b = new_sol.routes[move.route_b].nodes

        nodes_a[move.pos_u] = move.node_v
        nodes_b[move.pos_v] = move.node_u

        new_sol._rebuild_customer_map()
        return new_sol

    def undo(self, solution: 'Solution', move: SwapMove) -> 'Solution':
        """Undo Swap = Swap lại (symmetric operation)."""
        return self.apply(solution, move)


# ===========================================================================
# CROSS-EXCHANGE OPERATOR
# ===========================================================================

@dataclass
class CrossExchangeMove(Move):
    """
    Cross-exchange: Trao đổi đuôi của 2 routes từ vị trí cắt.
    
    Trước: route_a = [depot, A1, A2, | A3, A4, depot]
           route_b = [depot, B1, B2, | B3, B4, depot]
    
    Sau:   route_a = [depot, A1, A2, B3, B4, depot]
           route_b = [depot, B1, B2, A3, A4, depot]
    
    Đây là operator "aggressive" – thay đổi lớn, hữu ích để thoát local optima.
    """
    route_a: int = 0
    route_b: int = 0
    cut_a: int = 0  # Vị trí cắt trong route_a
    cut_b: int = 0  # Vị trí cắt trong route_b


class CrossExchangeOperator(BaseOperator):
    """
    Cross-exchange operator: trao đổi đuôi route giữa 2 xe.
    """

    def generate_moves(
        self,
        solution: 'Solution',
        problem: 'VRPProblem',
        max_moves: Optional[int] = None
    ) -> List[CrossExchangeMove]:
        moves = []
        dist = problem.dist_matrix
        routes = solution.routes
        n_routes = len(routes)

        for r_a in range(n_routes):
            route_a = routes[r_a]
            if len(route_a) < 2:
                continue
            nodes_a = route_a.nodes

            for r_b in range(r_a + 1, n_routes):
                route_b = routes[r_b]
                if len(route_b) < 2:
                    continue
                nodes_b = route_b.nodes

                for cut_a in range(1, len(nodes_a) - 1):
                    for cut_b in range(1, len(nodes_b) - 1):
                        a_pre = nodes_a[cut_a - 1]
                        a_post = nodes_a[cut_a]
                        b_pre = nodes_b[cut_b - 1]
                        b_post = nodes_b[cut_b]

                        old_cost = dist[a_pre, a_post] + dist[b_pre, b_post]
                        new_cost = dist[a_pre, b_post] + dist[b_pre, a_post]
                        delta = new_cost - old_cost

                        # Skip moves rõ ràng xấu
                        if delta > 50.0:
                            continue

                        move = CrossExchangeMove(
                            move_type='cross_exchange',
                            delta_cost=delta,
                            move_hash=hash((r_a, r_b, cut_a, cut_b, 'cross')),
                            route_a=r_a,
                            route_b=r_b,
                            cut_a=cut_a,
                            cut_b=cut_b,
                        )
                        moves.append(move)

                        if max_moves and len(moves) >= max_moves:
                            return moves

        return moves

    def apply(self, solution: 'Solution', move: CrossExchangeMove) -> 'Solution':
        """Trao đổi đuôi 2 routes từ vị trí cut."""
        new_sol = solution.copy()
        nodes_a = new_sol.routes[move.route_a].nodes
        nodes_b = new_sol.routes[move.route_b].nodes

        # Tách đầu và đuôi
        head_a = nodes_a[:move.cut_a]
        tail_a = nodes_a[move.cut_a:]
        head_b = nodes_b[:move.cut_b]
        tail_b = nodes_b[move.cut_b:]

        # Ghép lại đã trao đổi đuôi
        new_sol.routes[move.route_a]._nodes = head_a + tail_b
        new_sol.routes[move.route_b]._nodes = head_b + tail_a

        new_sol._rebuild_customer_map()
        return new_sol

    def undo(self, solution: 'Solution', move: CrossExchangeMove) -> 'Solution':
        """Undo Cross-exchange = thực hiện lại (symmetric)."""
        return self.apply(solution, move)