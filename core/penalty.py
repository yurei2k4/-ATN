"""
core/penalty.py
---------------
Dynamic Penalty Controller với Strategic Oscillation.

Chiến lược Dao động (Strategic Oscillation):
- Khi solver đang trong vùng feasible → tăng λ để tìm kiếm sâu hơn
- Khi solver đang trong vùng infeasible → giảm λ để kéo về feasible
- Cân bằng giữa Exploration (khám phá) và Exploitation (khai thác)

Hàm mục tiêu mở rộng:
    F(s) = C(s) + Σ λ_i * V_i(s)

Trong đó:
    C(s)  = tổng khoảng cách
    λ_i   = hệ số phạt cho ràng buộc i
    V_i(s)= mức độ vi phạm ràng buộc i
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np


class PenaltyController:
    """
    Quản lý hệ số phạt động λ cho tất cả các ràng buộc.
    
    Attributes:
        lambdas         : {constraint_name: λ_value}
        base_lambdas    : {constraint_name: λ_initial}
        update_freq     : Tần suất cập nhật λ (số iterations)
        feasible_ratio  : Tỷ lệ mục tiêu solutions feasible
        increase_factor : Hệ số tăng λ khi quá nhiều infeasible
        decrease_factor : Hệ số giảm λ khi quá nhiều feasible
    """

    DEFAULT_LAMBDAS = {
        'capacity': 1.0,
        'time_window': 1.0,
        'asymmetric': 0.5,
        'duration': 0.5,
    }

    def __init__(
        self,
        constraint_names: List[str],
        initial_lambda: float = 1.0,
        update_freq: int = 10,
        feasible_ratio_target: float = 0.5,
        increase_factor: float = 1.2,
        decrease_factor: float = 0.9,
        lambda_min: float = 0.1,
        lambda_max: float = 100.0,
    ):
        self.lambdas: Dict[str, float] = {
            name: initial_lambda for name in constraint_names
        }
        self.base_lambdas = self.lambdas.copy()
        self.update_freq = update_freq
        self.feasible_ratio_target = feasible_ratio_target
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # Lịch sử để tính feasible ratio
        self._recent_feasible: List[bool] = []
        self._window_size = update_freq * 2

        # Lịch sử λ để plot convergence
        self.lambda_history: Dict[str, List[float]] = {
            name: [initial_lambda] for name in constraint_names
        }

    def record_feasibility(self, is_feasible: bool):
        """Ghi nhận tính feasible của iteration hiện tại."""
        self._recent_feasible.append(is_feasible)
        if len(self._recent_feasible) > self._window_size:
            self._recent_feasible.pop(0)

    def update(self, iteration: int):
        """
        Cập nhật λ theo Strategic Oscillation.
        Được gọi mỗi `update_freq` iterations.
        """
        if iteration % self.update_freq != 0 or len(self._recent_feasible) < 2:
            return

        current_ratio = sum(self._recent_feasible) / len(self._recent_feasible)

        for name in self.lambdas:
            if current_ratio > self.feasible_ratio_target:
                # Quá nhiều solutions feasible → tăng λ để explore vùng infeasible
                # (tìm kiếm giải pháp tốt hơn ở ranh giới feasibility)
                self.lambdas[name] = min(
                    self.lambdas[name] * self.increase_factor,
                    self.lambda_max
                )
            else:
                # Quá nhiều solutions infeasible → giảm λ để kéo về feasible
                self.lambdas[name] = max(
                    self.lambdas[name] * self.decrease_factor,
                    self.lambda_min
                )
            self.lambda_history[name].append(self.lambdas[name])

    def get_lambda(self, constraint_name: str) -> float:
        """Lấy λ hiện tại của một ràng buộc."""
        return self.lambdas.get(constraint_name, 1.0)

    def get_all_lambdas(self) -> Dict[str, float]:
        """Lấy tất cả λ hiện tại."""
        return self.lambdas.copy()

    def reset(self):
        """Reset về λ ban đầu."""
        self.lambdas = self.base_lambdas.copy()
        self._recent_feasible.clear()

    def compute_penalty(self, violations: Dict[str, float]) -> float:
        """
        Tính tổng penalty: Σ λ_i * V_i
        
        Args:
            violations: {constraint_name: violation_value}
        """
        return sum(
            self.lambdas.get(name, 1.0) * value
            for name, value in violations.items()
        )

    def __repr__(self):
        lambdas_str = ', '.join(f'{k}={v:.3f}' for k, v in self.lambdas.items())
        return f"PenaltyController({lambdas_str})"