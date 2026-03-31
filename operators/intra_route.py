"""
operators/base.py & intra_route.py
-----------------------------------
Toán tử lân cận (Neighborhood Operators).

Mỗi operator tạo ra một "Move" – đối tượng mô tả sự thay đổi.
Move có thể được:
    - Áp dụng (apply)
    - Hoàn tác (undo) – dùng cho Undo/Redo trong Tabu Search
    - Tính Delta nhanh trước khi thực sự apply

Nhóm Intra-route (trong cùng một lộ trình):
    - 2-opt  : Đảo ngược một đoạn lộ trình
    - Or-opt : Di chuyển 1/2/3 node liên tiếp sang vị trí khác trong cùng route
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple
import numpy as np

if TYPE_CHECKING:
    from core.models import Solution, VRPProblem


# ===========================================================================
# BASE MOVE
# ===========================================================================

@dataclass
class Move:
    """
    Lớp cơ sở cho tất cả các moves.
    
    Attributes:
        move_type   : Tên loại move
        delta_cost  : Thay đổi chi phí ước tính (âm = cải thiện)
        move_hash   : Hash để kiểm tra tabu
        metadata    : Thông tin phụ cho Delta Evaluation và logging
    """
    move_type: str
    delta_cost: float = 0.0
    move_hash: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_improving(self) -> bool:
        return self.delta_cost < -1e-9


class BaseOperator(ABC):
    """
    Interface cho tất cả neighborhood operators.
    """

    @abstractmethod
    def generate_moves(
        self,
        solution: 'Solution',
        problem: 'VRPProblem',
        max_moves: Optional[int] = None
    ) -> List[Move]:
        """
        Sinh tất cả (hoặc tối đa max_moves) moves có thể từ solution hiện tại.
        """
        ...

    @abstractmethod
    def apply(self, solution: 'Solution', move: Move) -> 'Solution':
        """Áp dụng move vào solution, trả về solution mới."""
        ...

    @abstractmethod
    def undo(self, solution: 'Solution', move: Move) -> 'Solution':
        """Hoàn tác move."""
        ...


# ===========================================================================
# INTRA-ROUTE: 2-OPT
# ===========================================================================

@dataclass
class TwoOptMove(Move):
    """
    2-opt Move: Đảo ngược đoạn [i+1, j] trong route.
    
    Trước: ... → A → B → ... → C → D → ...
    Sau:   ... → A → C → ... → B → D → ...
    
    Tức là: xóa cạnh (A,B) và (C,D), thêm cạnh (A,C) và (B,D)
    """
    route_idx: int = 0
    i: int = 0  # Vị trí trước đoạn đảo ngược
    j: int = 0  # Vị trí cuối đoạn đảo ngược


class TwoOptOperator(BaseOperator):
    """
    2-opt operator cho intra-route improvement.
    Đây là operator hiệu quả nhất cho CVRP thuần túy.
    """

    def generate_moves(
        self,
        solution: 'Solution',
        problem: 'VRPProblem',
        max_moves: Optional[int] = None
    ) -> List[TwoOptMove]:
        moves = []
        dist = problem.dist_matrix

        for r_idx, route in enumerate(solution.routes):
            nodes = route.nodes
            n = len(nodes)
            if n < 4:  # Cần ít nhất 2 customers
                continue

            for i in range(1, n - 2):
                for j in range(i + 1, n - 1):
                    # Chi phí hiện tại: (nodes[i-1]→nodes[i]) + (nodes[j]→nodes[j+1])
                    a, b = nodes[i - 1], nodes[i]
                    c, d = nodes[j], nodes[j + 1]

                    current_cost = dist[a, b] + dist[c, d]
                    new_cost = dist[a, c] + dist[b, d]
                    delta = new_cost - current_cost

                    # Short-circuit: bỏ qua moves rõ ràng không cải thiện nhiều
                    if delta > 100.0:
                        continue

                    # Tính move_hash (đơn giản: dùng tuple hash)
                    move_hash = hash((r_idx, i, j, 'two_opt'))

                    move = TwoOptMove(
                        move_type='two_opt',
                        delta_cost=delta,
                        move_hash=move_hash,
                        route_idx=r_idx,
                        i=i,
                        j=j,
                        metadata={
                            'route_idx': r_idx,
                            'affected_nodes': nodes[i:j + 1],
                            'start_pos': i,
                        }
                    )
                    moves.append(move)

                    if max_moves and len(moves) >= max_moves:
                        return moves

        return moves

    def apply(self, solution: 'Solution', move: TwoOptMove) -> 'Solution':
        """
        Áp dụng 2-opt: đảo ngược đoạn [i, j] trong route.
        """
        new_sol = solution.copy()
        route = new_sol.routes[move.route_idx]
        nodes = route.nodes

        # Đảo ngược đoạn [i, j] (inclusive)
        nodes[move.i:move.j + 1] = nodes[move.i:move.j + 1][::-1]
        route._nodes = nodes

        new_sol._rebuild_customer_map()
        return new_sol

    def undo(self, solution: 'Solution', move: TwoOptMove) -> 'Solution':
        """Undo 2-opt = apply lại 2-opt (đảo ngược 2 lần = về ban đầu)."""
        return self.apply(solution, move)


# ===========================================================================
# INTRA-ROUTE: OR-OPT
# ===========================================================================

@dataclass
class OrOptMove(Move):
    """
    Or-opt Move: Di chuyển một chuỗi segment_size nodes liên tiếp
    sang vị trí khác trong cùng route.
    
    segment_size: 1, 2, hoặc 3 nodes
    """
    route_idx: int = 0
    from_pos: int = 0       # Vị trí bắt đầu của segment
    to_pos: int = 0         # Vị trí chèn vào (sau node to_pos)
    segment_size: int = 1   # Số nodes trong segment


class OrOptOperator(BaseOperator):
    """
    Or-opt operator: di chuyển 1/2/3 nodes trong cùng route.
    Hiệu quả để tinh chỉnh lộ trình sau 2-opt.
    """

    def __init__(self, segment_sizes: List[int] = None):
        self.segment_sizes = segment_sizes or [1, 2, 3]

    def generate_moves(
        self,
        solution: 'Solution',
        problem: 'VRPProblem',
        max_moves: Optional[int] = None
    ) -> List[OrOptMove]:
        moves = []
        dist = problem.dist_matrix

        for r_idx, route in enumerate(solution.routes):
            nodes = route.nodes
            n = len(nodes)

            for seg_size in self.segment_sizes:
                # from_pos: vị trí bắt đầu segment (1 đến n-seg_size-1)
                for from_pos in range(1, n - seg_size):
                    seg = nodes[from_pos:from_pos + seg_size]
                    prev_node = nodes[from_pos - 1]
                    next_node = nodes[from_pos + seg_size]

                    # Chi phí xóa segment khỏi vị trí hiện tại
                    removal_delta = (
                        - dist[prev_node, seg[0]]
                        - dist[seg[-1], next_node]
                        + dist[prev_node, next_node]
                    )

                    # Thử chèn vào vị trí khác
                    for to_pos in range(1, n - 1):
                        if to_pos in range(from_pos - 1, from_pos + seg_size + 1):
                            continue  # Vị trí giống, skip

                        insert_after = nodes[to_pos - 1]
                        insert_before = nodes[to_pos]

                        insertion_delta = (
                            - dist[insert_after, insert_before]
                            + dist[insert_after, seg[0]]
                            + dist[seg[-1], insert_before]
                        )

                        delta = removal_delta + insertion_delta

                        move = OrOptMove(
                            move_type=f'or_opt_{seg_size}',
                            delta_cost=delta,
                            move_hash=hash((r_idx, from_pos, to_pos, seg_size, 'or_opt')),
                            route_idx=r_idx,
                            from_pos=from_pos,
                            to_pos=to_pos,
                            segment_size=seg_size,
                            metadata={
                                'route_idx': r_idx,
                                'start_pos': min(from_pos, to_pos),
                            }
                        )
                        moves.append(move)

                        if max_moves and len(moves) >= max_moves:
                            return moves

        return moves

    def apply(self, solution: 'Solution', move: OrOptMove) -> 'Solution':
        """
        Áp dụng Or-opt:
        1. Xóa segment khỏi from_pos
        2. Chèn vào to_pos
        """
        new_sol = solution.copy()
        route = new_sol.routes[move.route_idx]
        nodes = route.nodes

        # Lấy segment ra
        seg = nodes[move.from_pos:move.from_pos + move.segment_size]
        del nodes[move.from_pos:move.from_pos + move.segment_size]

        # Điều chỉnh to_pos sau khi xóa
        actual_to = move.to_pos
        if move.to_pos > move.from_pos:
            actual_to -= move.segment_size

        # Chèn segment vào vị trí mới
        for k, node_id in enumerate(seg):
            nodes.insert(actual_to + k, node_id)

        route._nodes = nodes
        new_sol._rebuild_customer_map()
        return new_sol

    def undo(self, solution: 'Solution', move: OrOptMove) -> 'Solution':
        """
        Undo Or-opt: tạo reverse move và apply.
        """
        # Tạo reverse move: from_pos và to_pos hoán đổi
        reverse_move = OrOptMove(
            move_type=move.move_type,
            delta_cost=-move.delta_cost,
            route_idx=move.route_idx,
            from_pos=move.to_pos,
            to_pos=move.from_pos,
            segment_size=move.segment_size,
        )
        return self.apply(solution, reverse_move)