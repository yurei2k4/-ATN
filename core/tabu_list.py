"""
core/tabu_list.py
-----------------
Tabu List sử dụng Zobrist Hashing để kiểm tra trạng thái trong O(1).

Zobrist Hashing: Mỗi (node, position) được gán một số ngẫu nhiên 64-bit.
Hash của một move = XOR của các số này.
Kỹ thuật này cho phép kiểm tra tabu cực kỳ nhanh mà không cần lưu toàn bộ solution.
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import Tuple, Optional


class ZobristHasher:
    """
    Tính Zobrist Hash cho các moves.
    
    Mỗi cặp (node_id, route_idx) có một số random 64-bit duy nhất.
    Hash của một move = XOR(hash(node_i, old_pos), hash(node_i, new_pos), ...)
    """

    def __init__(self, num_nodes: int, num_routes: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        # Bảng hash: [node_id, route_idx] → số 64-bit ngẫu nhiên
        self._table = rng.integers(
            low=0,
            high=np.iinfo(np.int64).max,
            size=(num_nodes, num_routes),
            dtype=np.int64
        )

    def hash_node_in_route(self, node_id: int, route_idx: int) -> int:
        """Lấy hash value của một node tại một route cụ thể."""
        return int(self._table[node_id, route_idx])

    def hash_move(self, changes: list[Tuple[int, int, int, int]]) -> int:
        """
        Tính hash cho một move dựa trên các thay đổi.
        
        Args:
            changes: List of (node_id, old_route, new_route, _)
        
        Returns:
            Hash 64-bit của move này
        """
        h = 0
        for node_id, old_route, new_route, _ in changes:
            if old_route >= 0:
                h ^= self.hash_node_in_route(node_id, old_route)
            if new_route >= 0:
                h ^= self.hash_node_in_route(node_id, new_route)
        return h


class TabuList:
    """
    Tabu List với Zobrist Hashing.
    
    Attributes:
        tenure      : Số iteration một move bị cấm (Tabu Tenure)
        _tabu_dict  : {move_hash: expiry_iteration}
        _history    : Deque để theo dõi lịch sử (dùng cho Aspiration)
    """

    def __init__(self, tenure: int = 10, max_size: int = 500):
        self.tenure = tenure
        self.max_size = max_size
        # Hash → iteration hết hạn
        self._tabu_dict: dict[int, int] = {}
        # Deque lưu (iteration, hash) để dọn dẹp entries cũ
        self._expiry_queue: deque = deque()

    def add(self, move_hash: int, current_iteration: int):
        """Thêm một move vào tabu list."""
        expiry = current_iteration + self.tenure
        self._tabu_dict[move_hash] = expiry
        self._expiry_queue.append((expiry, move_hash))
        
        # Dọn dẹp nếu quá lớn
        if len(self._tabu_dict) > self.max_size:
            self._cleanup(current_iteration)

    def is_tabu(self, move_hash: int, current_iteration: int) -> bool:
        """Kiểm tra move có đang trong tabu list không."""
        expiry = self._tabu_dict.get(move_hash, -1)
        if expiry < 0:
            return False
        if current_iteration >= expiry:
            # Đã hết hạn, xóa luôn
            del self._tabu_dict[move_hash]
            return False
        return True

    def _cleanup(self, current_iteration: int):
        """Xóa các entries đã hết hạn."""
        while self._expiry_queue:
            expiry, h = self._expiry_queue[0]
            if current_iteration >= expiry:
                self._expiry_queue.popleft()
                if self._tabu_dict.get(h) == expiry:
                    del self._tabu_dict[h]
            else:
                break

    def update_tenure(self, new_tenure: int):
        """Điều chỉnh tabu tenure (dùng bởi Strategic Oscillation)."""
        self.tenure = max(1, new_tenure)

    def clear(self):
        """Xóa toàn bộ tabu list."""
        self._tabu_dict.clear()
        self._expiry_queue.clear()

    def __len__(self):
        return len(self._tabu_dict)

    def __repr__(self):
        return f"TabuList(tenure={self.tenure}, size={len(self)})"


class AspirationCriteria:
    """
    Aspiration Criteria: Cho phép chấp nhận một move tabu nếu
    nó dẫn đến giải pháp tốt hơn best solution toàn cục.
    
    Đây là cơ chế escape khỏi local optima khi tabu list quá restrictive.
    """

    def __init__(self):
        self.best_cost: float = float('inf')

    def update_best(self, cost: float):
        """Cập nhật chi phí tốt nhất đã biết."""
        if cost < self.best_cost:
            self.best_cost = cost

    def is_aspired(self, candidate_cost: float) -> bool:
        """
        Trả về True nếu candidate tốt hơn best → override tabu.
        
        Args:
            candidate_cost: Chi phí của candidate solution
        """
        return candidate_cost < self.best_cost

    def reset(self):
        self.best_cost = float('inf')