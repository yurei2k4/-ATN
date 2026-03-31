"""
core/solver.py
--------------
UTS Core Engine – Unified Tabu Search Solver.

Đây là "Lõi toán học" (Solver) được tách biệt hoàn toàn khỏi
"Logic nghiệp vụ" (Plugins) theo nguyên lý IoC.

Luồng thuật toán UTS:
    1. Khởi tạo giải pháp ban đầu (greedy nearest neighbor)
    2. Lặp cho đến khi đạt điều kiện dừng:
       a. Sinh các moves từ tất cả operators
       b. Sắp xếp theo delta_cost
       c. Chọn move tốt nhất KHÔNG tabu (hoặc thỏa aspiration)
       d. Áp dụng move (kể cả nếu làm xấu hơn – relaxation)
       e. Cập nhật Tabu List
       f. Cập nhật Best Solution nếu feasible + tốt hơn
       g. Cập nhật λ theo Strategic Oscillation
    3. Trả về best feasible solution
"""

from __future__ import annotations
import time
import logging
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass

import numpy as np

from core.models import VRPProblem, Solution, Route
from core.tabu_list import TabuList, ZobristHasher, AspirationCriteria
from core.penalty import PenaltyController
from plugins.base import PluginRegistry
from operators.intra_route import TwoOptOperator, OrOptOperator, Move
from operators.inter_route import RelocateOperator, SwapOperator, CrossExchangeOperator

logger = logging.getLogger(__name__)


# ===========================================================================
# SOLVER CONFIG
# ===========================================================================

@dataclass
class UTSConfig:
    """
    Cấu hình thuật toán UTS.
    
    Attributes:
        max_iterations      : Số vòng lặp tối đa
        max_time_seconds    : Thời gian chạy tối đa (giây)
        tabu_tenure         : Độ dài tabu (số iterations một move bị cấm)
        tabu_tenure_factor  : Hệ số tính tenure động (tenure = factor * sqrt(n))
        max_no_improve      : Dừng sớm nếu không cải thiện sau N iterations
        penalty_update_freq : Tần suất cập nhật λ
        feasible_ratio_target: Tỷ lệ mục tiêu solutions feasible
        max_moves_per_iter  : Giới hạn moves sinh ra mỗi iteration (performance)
        verbose             : In log chi tiết
        random_seed         : Seed cho reproducibility
    """
    max_iterations: int = 1000
    max_time_seconds: float = 60.0
    tabu_tenure: Optional[int] = None   # None = tính tự động
    tabu_tenure_factor: float = 0.5     # tenure = factor * sqrt(n_customers)
    max_no_improve: int = 200
    penalty_update_freq: int = 10
    feasible_ratio_target: float = 0.5
    max_moves_per_iter: Optional[int] = None  # None = không giới hạn
    verbose: bool = True
    log_interval: int = 50
    random_seed: int = 42


# ===========================================================================
# CONSTRUCTION HEURISTIC
# ===========================================================================

def greedy_nearest_neighbor(problem: VRPProblem) -> Solution:
    """
    Thuật toán khởi tạo Greedy Nearest Neighbor.
    
    Xây dựng giải pháp ban đầu:
    - Lần lượt giao mỗi khách hàng cho xe gần nhất có đủ capacity
    - Nếu không xe nào nhận được → tạo xe mới (nếu còn)
    
    Returns:
        Solution ban đầu (có thể infeasible nhưng đủ để bắt đầu search)
    """
    solution = Solution(problem)
    unvisited = set(c.id for c in problem.customers)
    dist = problem.dist_matrix

    # Tạo routes cho tất cả vehicles
    routes = []
    for vehicle in problem.vehicles:
        route = Route(vehicle, depot_id=problem.depot.id)
        routes.append(route)
        solution.add_route(route)

    vehicle_idx = 0
    route = solution.routes[vehicle_idx]
    current_node = problem.depot.id

    while unvisited:
        # Tìm customer gần nhất chưa thăm
        best_next = None
        best_dist = float('inf')

        for cid in unvisited:
            d = dist[current_node, cid]
            if d < best_dist:
                # Kiểm tra capacity sơ bộ
                if route.total_load(problem) + problem.nodes[cid].demand <= route.vehicle.capacity + 1e-6:
                    best_dist = d
                    best_next = cid

        if best_next is None:
            # Không thêm được vào route hiện tại → chuyển xe mới
            vehicle_idx += 1
            if vehicle_idx >= len(solution.routes):
                # Hết xe → thêm customer vào route cuối (infeasible, solver sẽ sửa)
                vehicle_idx = len(solution.routes) - 1
                # Thêm vào route gần nhất
                best_next = min(
                    unvisited,
                    key=lambda cid: dist[current_node, cid]
                )
            route = solution.routes[vehicle_idx]
            current_node = problem.depot.id

        route.append_customer(best_next)
        unvisited.remove(best_next)
        current_node = best_next

    return solution


# ===========================================================================
# MAIN UTS SOLVER
# ===========================================================================

class UTSSolver:
    """
    Unified Tabu Search Solver.
    
    Tích hợp:
        - Tabu List với Zobrist Hashing
        - Dynamic Penalty Controller (Strategic Oscillation)
        - Plugin Registry (IoC)
        - Tất cả Neighborhood Operators
        - Aspiration Criteria
        - Convergence tracking
    """

    def __init__(
        self,
        problem: VRPProblem,
        config: UTSConfig = None,
        plugin_registry: PluginRegistry = None,
    ):
        self.problem = problem
        self.config = config or UTSConfig()
        self.registry = plugin_registry or PluginRegistry()

        # Tính tabu tenure tự động nếu không được chỉ định
        n = len(problem.customers)
        if self.config.tabu_tenure is None:
            tenure = max(5, int(self.config.tabu_tenure_factor * np.sqrt(n)))
        else:
            tenure = self.config.tabu_tenure

        # Khởi tạo components
        self.tabu_list = TabuList(tenure=tenure)
        self.hasher = ZobristHasher(
            num_nodes=problem.num_nodes,
            num_routes=problem.num_vehicles,
            seed=self.config.random_seed
        )
        self.aspiration = AspirationCriteria()

        constraint_names = [p.name for p in self.registry.all()]
        self.penalty_ctrl = PenaltyController(
            constraint_names=constraint_names,
            update_freq=self.config.penalty_update_freq,
            feasible_ratio_target=self.config.feasible_ratio_target,
        )

        # Operators
        self.operators = [
            TwoOptOperator(),
            OrOptOperator(segment_sizes=[1, 2, 3]),
            RelocateOperator(),
            SwapOperator(),
            CrossExchangeOperator(),
        ]

        # Tracking
        self.iteration = 0
        self.best_solution: Optional[Solution] = None
        self.best_cost: float = float('inf')
        self.convergence_history: List[Dict[str, float]] = []
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Main Solve
    # ------------------------------------------------------------------

    def solve(self, initial_solution: Optional[Solution] = None) -> Solution:
        """
        Entry point chính của solver.
        
        Args:
            initial_solution: Giải pháp khởi đầu (nếu None → dùng greedy)
        
        Returns:
            Best feasible solution tìm được
        """
        self._start_time = time.time()
        np.random.seed(self.config.random_seed)

        # Khởi tạo
        if initial_solution is None:
            current_sol = greedy_nearest_neighbor(self.problem)
            logger.info(f"Greedy khởi tạo: dist={current_sol.total_distance():.2f}")
        else:
            current_sol = initial_solution.copy()

        self.best_solution = current_sol.copy()
        self.best_cost = self._augmented_cost(current_sol)
        self.aspiration.update_best(self.best_cost)

        # Thông báo plugins
        for plugin in self.registry.all():
            plugin.on_solution_start(current_sol)

        no_improve_count = 0

        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"UTS Solver bắt đầu: {self.problem.name}")
            print(f"  Nodes: {self.problem.num_nodes}, Vehicles: {self.problem.num_vehicles}")
            print(f"  Plugins: {[p.name for p in self.registry.all()]}")
            print(f"  Tabu tenure: {self.tabu_list.tenure}")
            print(f"  Max iterations: {self.config.max_iterations}")
            print(f"{'='*60}\n")

        # ----------------------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------------------
        for iteration in range(1, self.config.max_iterations + 1):
            self.iteration = iteration

            # Kiểm tra điều kiện dừng
            elapsed = time.time() - self._start_time
            if elapsed >= self.config.max_time_seconds:
                logger.info(f"Dừng sớm: hết thời gian ({elapsed:.1f}s)")
                break
            if no_improve_count >= self.config.max_no_improve:
                logger.info(f"Dừng sớm: không cải thiện sau {no_improve_count} iterations")
                break

            # Sinh tất cả moves
            all_moves = self._generate_all_moves(
                current_sol,
                max_moves=self.config.max_moves_per_iter
            )

            if not all_moves:
                break

            # Sắp xếp theo delta_cost (tốt nhất trước)
            all_moves.sort(key=lambda m: m.delta_cost)

            # Chọn best non-tabu move (hoặc aspired tabu move)
            selected_move = self._select_move(all_moves, current_sol, iteration)

            if selected_move is None:
                # Không tìm được move phù hợp → diversification
                selected_move = all_moves[0]

            # Áp dụng move
            new_sol = self._apply_move(current_sol, selected_move)

            # Tính cost mới
            new_cost = self._augmented_cost(new_sol)
            is_feasible = self._is_feasible(new_sol)

            # Cập nhật tabu list
            self.tabu_list.add(selected_move.move_hash, iteration)

            # Cập nhật best solution
            feasible_cost = new_sol.total_distance() if is_feasible else float('inf')
            if is_feasible and feasible_cost < self.best_cost:
                self.best_solution = new_sol.copy()
                self.best_cost = feasible_cost
                self.aspiration.update_best(self.best_cost)
                no_improve_count = 0
                if self.config.verbose and iteration % 10 == 0:
                    print(f"  ✓ Iter {iteration:4d}: Best={self.best_cost:.2f} "
                          f"(feasible, {self._vehicles_used(new_sol)} xe)")
            else:
                no_improve_count += 1

            # Cập nhật penalty controller
            self.penalty_ctrl.record_feasibility(is_feasible)
            self.penalty_ctrl.update(iteration)

            # Thông báo plugins về move được chấp nhận
            for plugin in self.registry.all():
                plugin.on_move_accepted(selected_move.metadata)

            # Move đến solution mới (luôn accept – relaxation strategy)
            current_sol = new_sol

            # Log định kỳ
            if self.config.verbose and iteration % self.config.log_interval == 0:
                elapsed = time.time() - self._start_time
                print(f"  Iter {iteration:4d}/{self.config.max_iterations} | "
                      f"Current={new_cost:.2f} | Best={self.best_cost:.2f} | "
                      f"Tabu={len(self.tabu_list)} | "
                      f"Time={elapsed:.1f}s | "
                      f"λ={list(self.penalty_ctrl.lambdas.values())[0]:.3f}")

            # Lưu convergence history
            self.convergence_history.append({
                'iteration': iteration,
                'current_cost': new_sol.total_distance(),
                'best_cost': self.best_cost,
                'augmented_cost': new_cost,
                'is_feasible': is_feasible,
                'elapsed': time.time() - self._start_time,
                **{f'lambda_{k}': v for k, v in self.penalty_ctrl.lambdas.items()},
            })

        # ----------------------------------------------------------------
        # KẾT QUẢ
        # ----------------------------------------------------------------
        total_time = time.time() - self._start_time
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"UTS hoàn thành sau {self.iteration} iterations ({total_time:.2f}s)")
            if self.best_solution:
                print(f"  Best distance: {self.best_cost:.2f}")
                print(f"  Vehicles used: {self._vehicles_used(self.best_solution)}")
            print(f"{'='*60}\n")

        return self.best_solution or current_sol

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _generate_all_moves(
        self,
        solution: Solution,
        max_moves: Optional[int] = None
    ) -> List[Move]:
        """Sinh moves từ tất cả operators."""
        all_moves = []
        per_operator = (max_moves // len(self.operators)) if max_moves else None

        for op in self.operators:
            moves = op.generate_moves(solution, self.problem, max_moves=per_operator)
            all_moves.extend(moves)

        return all_moves

    def _select_move(
        self,
        sorted_moves: List[Move],
        current_sol: Solution,
        iteration: int
    ) -> Optional[Move]:
        """
        Chọn move tốt nhất:
        - Không tabu, HOẶC
        - Tabu nhưng thỏa aspiration criteria
        """
        for move in sorted_moves:
            is_tabu = self.tabu_list.is_tabu(move.move_hash, iteration)

            if not is_tabu:
                return move

            # Aspiration: override tabu nếu move này dẫn đến best solution
            if is_tabu:
                estimated_cost = self.best_cost + move.delta_cost
                if self.aspiration.is_aspired(estimated_cost):
                    logger.debug(f"Aspiration override tại iter {iteration}")
                    return move

        return None

    def _apply_move(self, solution: Solution, move: Move) -> Solution:
        """Áp dụng move bằng operator tương ứng."""
        move_type = move.move_type

        for op in self.operators:
            if hasattr(op, 'generate_moves'):
                # Xác định operator phù hợp theo move type
                if move_type == 'two_opt' and isinstance(op, TwoOptOperator):
                    return op.apply(solution, move)
                elif move_type.startswith('or_opt') and isinstance(op, OrOptOperator):
                    return op.apply(solution, move)
                elif move_type == 'relocate' and isinstance(op, RelocateOperator):
                    return op.apply(solution, move)
                elif move_type == 'swap' and isinstance(op, SwapOperator):
                    return op.apply(solution, move)
                elif move_type == 'cross_exchange' and isinstance(op, CrossExchangeOperator):
                    return op.apply(solution, move)

        # Fallback: không tìm thấy operator → trả lại solution cũ
        logger.warning(f"Không tìm thấy operator cho move_type='{move_type}'")
        return solution.copy()

    def _augmented_cost(self, solution: Solution) -> float:
        """
        F(s) = C(s) + Σ λ_i * V_i(s)
        """
        dist_cost = solution.total_distance()
        violations = {}
        for plugin in self.registry.all():
            total_viol = sum(
                plugin.compute_violation(route, self.problem)
                for route in solution.routes
            )
            violations[plugin.name] = total_viol

        penalty = self.penalty_ctrl.compute_penalty(violations)
        return dist_cost + penalty

    def _is_feasible(self, solution: Solution) -> bool:
        """Kiểm tra solution có feasible theo tất cả plugins."""
        for plugin in self.registry.all():
            for route in solution.routes:
                if plugin.compute_violation(route, self.problem) > 1e-6:
                    return False
        return True

    def _vehicles_used(self, solution: Solution) -> int:
        return sum(1 for r in solution.routes if not r.is_empty)

    def get_convergence_df(self):
        """Xuất convergence history dưới dạng DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(self.convergence_history)
        except ImportError:
            return self.convergence_history

    def get_stats(self) -> Dict[str, Any]:
        """Thống kê sau khi solve xong."""
        if not self.best_solution:
            return {}
        return {
            'best_distance': self.best_cost,
            'vehicles_used': self._vehicles_used(self.best_solution),
            'iterations': self.iteration,
            'total_time': time.time() - self._start_time,
            'tabu_tenure': self.tabu_list.tenure,
            'final_lambdas': self.penalty_ctrl.get_all_lambdas(),
        }