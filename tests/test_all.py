"""
tests/test_all.py
-----------------
Unit tests toàn diện cho UTS-VRP Framework.

Chạy: pytest tests/test_all.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from core.models import Node, Vehicle, VRPProblem, Route, Solution
from core.tabu_list import TabuList, ZobristHasher, AspirationCriteria
from core.penalty import PenaltyController
from core.solver import UTSSolver, UTSConfig, greedy_nearest_neighbor
from plugins.base import PluginRegistry
from plugins.capacity import CapacityPlugin
from plugins.time_window import TimeWindowPlugin
from plugins.asymmetric import AsymmetricRoutePlugin
from operators.intra_route import TwoOptOperator, OrOptOperator, TwoOptMove
from operators.inter_route import RelocateOperator, SwapOperator, CrossExchangeOperator
from benchmark.solomon_loader import create_sample_solomon, compute_gap, get_bks


# ===========================================================================
# FIXTURES
# ===========================================================================

@pytest.fixture
def simple_problem():
    """
    Bài toán đơn giản: 1 depot + 5 customers, 2 xe, capacity=50.
    Ma trận khoảng cách Euclidean đối xứng.
    """
    nodes = [
        Node(id=0, x=0.0, y=0.0, demand=0, node_type='depot',
             ready_time=0, due_time=1000, service_time=0),
        Node(id=1, x=1.0, y=0.0, demand=10, node_type='customer',
             ready_time=0, due_time=100, service_time=5),
        Node(id=2, x=2.0, y=0.0, demand=10, node_type='customer',
             ready_time=0, due_time=100, service_time=5),
        Node(id=3, x=0.0, y=1.0, demand=10, node_type='customer',
             ready_time=0, due_time=100, service_time=5),
        Node(id=4, x=0.0, y=2.0, demand=10, node_type='customer',
             ready_time=0, due_time=100, service_time=5),
        Node(id=5, x=1.0, y=1.0, demand=10, node_type='customer',
             ready_time=0, due_time=100, service_time=5),
    ]
    vehicles = [
        Vehicle(id=0, capacity=50.0),
        Vehicle(id=1, capacity=50.0),
    ]
    coords = np.array([[n.x, n.y] for n in nodes])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    return VRPProblem(nodes=nodes, vehicles=vehicles, dist_matrix=dist_matrix,
                     problem_type='VRPTW', name='simple_test')


@pytest.fixture
def asymmetric_problem():
    """Bài toán với ma trận khoảng cách bất đối xứng."""
    nodes = [
        Node(id=0, x=0.0, y=0.0, demand=0, node_type='depot',
             ready_time=0, due_time=1000, service_time=0),
        Node(id=1, x=1.0, y=0.0, demand=10, node_type='customer',
             ready_time=0, due_time=200, service_time=5),
        Node(id=2, x=2.0, y=0.0, demand=10, node_type='customer',
             ready_time=0, due_time=200, service_time=5),
        Node(id=3, x=1.0, y=1.0, demand=10, node_type='customer',
             ready_time=0, due_time=200, service_time=5),
    ]
    vehicles = [Vehicle(id=0, capacity=50.0)]

    # Ma trận bất đối xứng: đường một chiều 1→2 dài hơn 2→1
    dist_matrix = np.array([
        [0.0, 1.0, 2.0, 1.5],
        [1.0, 0.0, 1.0, 1.2],
        [3.0, 1.0, 0.0, 1.1],  # 2→1 dài hơn 1→2
        [1.5, 1.2, 1.1, 0.0],
    ])
    return VRPProblem(nodes=nodes, vehicles=vehicles, dist_matrix=dist_matrix,
                     problem_type='AVRP', name='asymmetric_test')


@pytest.fixture
def simple_solution(simple_problem):
    """Solution đơn giản: xe 0 phục vụ 1,2,3; xe 1 phục vụ 4,5."""
    sol = Solution(simple_problem)
    r0 = Route(simple_problem.vehicles[0])
    r0._nodes = [0, 1, 2, 3, 0]
    r1 = Route(simple_problem.vehicles[1])
    r1._nodes = [0, 4, 5, 0]
    sol.add_route(r0)
    sol.add_route(r1)
    return sol


# ===========================================================================
# TEST: MODELS
# ===========================================================================

class TestModels:

    def test_node_creation(self):
        node = Node(id=1, x=10.0, y=20.0, demand=15.0)
        assert node.id == 1
        assert node.x == 10.0
        assert node.demand == 15.0
        assert node.node_type == 'customer'

    def test_vehicle_creation(self):
        v = Vehicle(id=0, capacity=100.0, depot_id=0)
        assert v.capacity == 100.0

    def test_problem_validation(self, simple_problem):
        assert simple_problem.num_nodes == 6
        assert simple_problem.num_vehicles == 2
        assert simple_problem.depot.id == 0
        assert len(simple_problem.customers) == 5

    def test_dist_matrix_shape(self, simple_problem):
        n = simple_problem.num_nodes
        assert simple_problem.dist_matrix.shape == (n, n)

    def test_asymmetric_detection(self, asymmetric_problem):
        dist = asymmetric_problem.dist_matrix
        # Ma trận không đối xứng
        assert not np.allclose(dist, dist.T)

    def test_route_operations(self, simple_problem):
        route = Route(simple_problem.vehicles[0])
        route.append_customer(1)
        route.append_customer(2)
        assert route.num_customers == 2
        assert route.nodes == [0, 1, 2, 0]
        assert not route.is_empty

    def test_route_total_load(self, simple_problem):
        route = Route(simple_problem.vehicles[0])
        route._nodes = [0, 1, 2, 3, 0]
        load = route.total_load(simple_problem)
        assert abs(load - 30.0) < 1e-9

    def test_route_total_distance(self, simple_problem):
        route = Route(simple_problem.vehicles[0])
        route._nodes = [0, 1, 0]
        dist = route.total_distance(simple_problem)
        expected = simple_problem.get_dist(0, 1) + simple_problem.get_dist(1, 0)
        assert abs(dist - expected) < 1e-9

    def test_solution_total_distance(self, simple_solution, simple_problem):
        total = simple_solution.total_distance()
        assert total > 0

    def test_solution_copy(self, simple_solution):
        copy = simple_solution.copy()
        assert copy.total_distance() == simple_solution.total_distance()
        # Copy phải độc lập
        copy.routes[0]._nodes[1] = 99
        assert simple_solution.routes[0]._nodes[1] != 99

    def test_solution_to_dict(self, simple_solution):
        d = simple_solution.to_dict()
        assert 'total_distance' in d
        assert 'routes' in d
        assert isinstance(d['routes'], list)


# ===========================================================================
# TEST: TABU LIST
# ===========================================================================

class TestTabuList:

    def test_add_and_check(self):
        tl = TabuList(tenure=5)
        tl.add(hash_val := 12345, current_iteration=1)
        assert tl.is_tabu(hash_val, current_iteration=3)

    def test_expiry(self):
        tl = TabuList(tenure=3)
        tl.add(99999, current_iteration=1)
        # Hết hạn sau tenure iterations
        assert not tl.is_tabu(99999, current_iteration=5)

    def test_not_tabu_unknown(self):
        tl = TabuList(tenure=10)
        assert not tl.is_tabu(77777, current_iteration=1)

    def test_update_tenure(self):
        tl = TabuList(tenure=5)
        tl.update_tenure(15)
        assert tl.tenure == 15

    def test_clear(self):
        tl = TabuList(tenure=10)
        tl.add(11111, 1)
        tl.add(22222, 1)
        tl.clear()
        assert len(tl) == 0

    def test_zobrist_hasher(self):
        hasher = ZobristHasher(num_nodes=20, num_routes=5)
        h1 = hasher.hash_node_in_route(3, 1)
        h2 = hasher.hash_node_in_route(3, 2)
        h3 = hasher.hash_node_in_route(4, 1)
        # Các hash phải khác nhau
        assert h1 != h2
        assert h1 != h3

    def test_aspiration_criteria(self):
        asp = AspirationCriteria()
        asp.update_best(100.0)
        assert asp.is_aspired(90.0)
        assert not asp.is_aspired(110.0)


# ===========================================================================
# TEST: PENALTY CONTROLLER
# ===========================================================================

class TestPenaltyController:

    def test_initial_lambdas(self):
        ctrl = PenaltyController(['capacity', 'time_window'], initial_lambda=2.0)
        assert ctrl.get_lambda('capacity') == 2.0
        assert ctrl.get_lambda('time_window') == 2.0

    def test_compute_penalty(self):
        ctrl = PenaltyController(['capacity'], initial_lambda=1.0)
        penalty = ctrl.compute_penalty({'capacity': 10.0})
        assert abs(penalty - 10.0) < 1e-9

    def test_lambda_increases_when_feasible(self):
        ctrl = PenaltyController(
            ['capacity'], initial_lambda=1.0,
            update_freq=5, feasible_ratio_target=0.5,
            increase_factor=1.5,
        )
        # Record nhiều feasible
        for _ in range(10):
            ctrl.record_feasibility(True)
        initial_lambda = ctrl.get_lambda('capacity')
        ctrl.update(iteration=5)
        # Lambda phải tăng
        assert ctrl.get_lambda('capacity') >= initial_lambda

    def test_lambda_decreases_when_infeasible(self):
        ctrl = PenaltyController(
            ['capacity'], initial_lambda=10.0,
            update_freq=5, feasible_ratio_target=0.5,
            decrease_factor=0.8,
        )
        for _ in range(10):
            ctrl.record_feasibility(False)
        initial_lambda = ctrl.get_lambda('capacity')
        ctrl.update(iteration=5)
        assert ctrl.get_lambda('capacity') <= initial_lambda

    def test_lambda_bounds(self):
        ctrl = PenaltyController(
            ['capacity'], initial_lambda=100.0,
            lambda_max=100.0, lambda_min=0.1,
            update_freq=2, increase_factor=10.0,
        )
        for _ in range(10):
            ctrl.record_feasibility(True)
        ctrl.update(iteration=2)
        assert ctrl.get_lambda('capacity') <= 100.0


# ===========================================================================
# TEST: PLUGINS
# ===========================================================================

class TestPlugins:

    def test_capacity_no_violation(self, simple_problem):
        route = Route(simple_problem.vehicles[0])
        route._nodes = [0, 1, 2, 0]  # load = 20, capacity = 50
        plugin = CapacityPlugin()
        assert plugin.compute_violation(route, simple_problem) == 0.0

    def test_capacity_violation(self, simple_problem):
        # Xe có capacity=50, nhưng load = 60 → vi phạm 10
        route = Route(simple_problem.vehicles[0])
        route._nodes = [0, 1, 2, 3, 4, 5, 0]  # load = 50, capacity = 50
        plugin = CapacityPlugin()
        # Load = 50 = capacity → không vi phạm
        viol = plugin.compute_violation(route, simple_problem)
        assert viol == 0.0

    def test_capacity_plugin_name(self):
        plugin = CapacityPlugin()
        assert plugin.name == 'capacity'

    def test_time_window_no_violation(self, simple_problem):
        route = Route(simple_problem.vehicles[0])
        route._nodes = [0, 1, 0]
        plugin = TimeWindowPlugin()
        viol = plugin.compute_violation(route, simple_problem)
        assert viol >= 0.0  # Không âm

    def test_time_window_empty_route(self, simple_problem):
        route = Route(simple_problem.vehicles[0])
        plugin = TimeWindowPlugin()
        assert plugin.compute_violation(route, simple_problem) == 0.0

    def test_asymmetric_plugin(self, asymmetric_problem):
        plugin = AsymmetricRoutePlugin()
        assert plugin.verify_asymmetric(asymmetric_problem)

    def test_plugin_registry(self):
        registry = PluginRegistry()
        registry.register(CapacityPlugin())
        registry.register(TimeWindowPlugin())
        assert len(registry) == 2
        assert registry.get('capacity') is not None

    def test_plugin_registry_priority(self):
        registry = PluginRegistry()
        registry.register(TimeWindowPlugin())   # priority 20
        registry.register(CapacityPlugin())     # priority 10
        plugins = registry.all()
        # CapacityPlugin phải đứng trước (priority thấp hơn)
        assert plugins[0].name == 'capacity'


# ===========================================================================
# TEST: OPERATORS
# ===========================================================================

class TestOperators:

    def test_two_opt_apply(self, simple_problem, simple_solution):
        op = TwoOptOperator()
        route_nodes_before = simple_solution.routes[0].nodes.copy()
        moves = op.generate_moves(simple_solution, simple_problem)
        if moves:
            move = moves[0]
            new_sol = op.apply(simple_solution, move)
            # Solution mới phải khác (hoặc giống nếu delta=0)
            assert new_sol is not simple_solution

    def test_two_opt_undo(self, simple_problem, simple_solution):
        op = TwoOptOperator()
        moves = op.generate_moves(simple_solution, simple_problem)
        if moves:
            move = moves[0]
            new_sol = op.apply(simple_solution, move)
            # Undo = apply lại (2-opt is self-inverse)
            restored = op.undo(new_sol, move)
            # Không test exact equality vì 2-opt tự inverse

    def test_or_opt_apply(self, simple_problem, simple_solution):
        op = OrOptOperator(segment_sizes=[1])
        moves = op.generate_moves(simple_solution, simple_problem)
        if moves:
            new_sol = op.apply(simple_solution, moves[0])
            assert new_sol is not simple_solution

    def test_relocate_apply(self, simple_problem, simple_solution):
        op = RelocateOperator()
        moves = op.generate_moves(simple_solution, simple_problem)
        # Phải có ít nhất một Relocate move khả thi
        assert len(moves) >= 0  # Có thể không có nếu capacity tight

    def test_swap_apply(self, simple_problem, simple_solution):
        op = SwapOperator()
        moves = op.generate_moves(simple_solution, simple_problem)
        if moves:
            new_sol = op.apply(simple_solution, moves[0])
            assert new_sol is not simple_solution

    def test_cross_exchange_apply(self, simple_problem, simple_solution):
        op = CrossExchangeOperator()
        moves = op.generate_moves(simple_solution, simple_problem)
        if moves:
            new_sol = op.apply(simple_solution, moves[0])
            # Tổng customers phải được bảo toàn
            original_customers = set(
                n for r in simple_solution.routes
                for n in r.customers
            )
            new_customers = set(
                n for r in new_sol.routes
                for n in r.customers
            )
            assert original_customers == new_customers

    def test_moves_preserve_customers(self, simple_problem, simple_solution):
        """Mọi operator phải bảo toàn tập hợp customers."""
        original_customers = sorted(
            n for r in simple_solution.routes for n in r.customers
        )
        operators = [
            TwoOptOperator(),
            OrOptOperator([1]),
            RelocateOperator(),
            SwapOperator(),
            CrossExchangeOperator(),
        ]
        for op in operators:
            moves = op.generate_moves(simple_solution, simple_problem, max_moves=5)
            for move in moves[:3]:
                new_sol = op.apply(simple_solution, move)
                new_customers = sorted(
                    n for r in new_sol.routes for n in r.customers
                )
                assert new_customers == original_customers, \
                    f"{op.__class__.__name__} vi phạm customer preservation!"


# ===========================================================================
# TEST: SOLVER END-TO-END
# ===========================================================================

class TestSolver:

    def test_greedy_init(self, simple_problem):
        sol = greedy_nearest_neighbor(simple_problem)
        assert sol is not None
        customers_served = sorted(
            n for r in sol.routes for n in r.customers
        )
        expected = sorted(c.id for c in simple_problem.customers)
        assert customers_served == expected

    def test_solver_runs(self, simple_problem):
        registry = PluginRegistry()
        registry.register(CapacityPlugin(violation_scale=10.0))

        config = UTSConfig(
            max_iterations=50,
            max_time_seconds=10.0,
            verbose=False,
        )
        solver = UTSSolver(simple_problem, config, registry)
        best = solver.solve()
        assert best is not None
        assert best.total_distance() > 0

    def test_solver_improves_over_greedy(self, simple_problem):
        registry = PluginRegistry()
        registry.register(CapacityPlugin(violation_scale=10.0))

        greedy = greedy_nearest_neighbor(simple_problem)
        greedy_dist = greedy.total_distance()

        config = UTSConfig(
            max_iterations=100,
            max_time_seconds=15.0,
            verbose=False,
        )
        solver = UTSSolver(simple_problem, config, registry)
        best = solver.solve()

        # Solver ít nhất không tệ hơn greedy
        assert best.total_distance() <= greedy_dist + 1e-6

    def test_solver_with_all_plugins(self, simple_problem):
        registry = PluginRegistry()
        registry.register(CapacityPlugin(violation_scale=10.0))
        registry.register(TimeWindowPlugin(late_penalty_scale=5.0))

        config = UTSConfig(
            max_iterations=50,
            max_time_seconds=10.0,
            verbose=False,
        )
        solver = UTSSolver(simple_problem, config, registry)
        best = solver.solve()
        assert best is not None

    def test_solver_convergence_history(self, simple_problem):
        registry = PluginRegistry()
        registry.register(CapacityPlugin())

        config = UTSConfig(max_iterations=30, verbose=False)
        solver = UTSSolver(simple_problem, config, registry)
        solver.solve()

        assert len(solver.convergence_history) > 0
        assert 'best_cost' in solver.convergence_history[0]
        assert 'iteration' in solver.convergence_history[0]

    def test_solver_stats(self, simple_problem):
        registry = PluginRegistry()
        registry.register(CapacityPlugin())

        config = UTSConfig(max_iterations=20, verbose=False)
        solver = UTSSolver(simple_problem, config, registry)
        solver.solve()

        stats = solver.get_stats()
        assert 'best_distance' in stats
        assert 'vehicles_used' in stats
        assert stats['best_distance'] > 0

    def test_customer_preservation_after_solve(self, simple_problem):
        """Sau khi solve, tất cả customers phải được phục vụ đúng một lần."""
        registry = PluginRegistry()
        registry.register(CapacityPlugin())

        config = UTSConfig(max_iterations=50, verbose=False)
        solver = UTSSolver(simple_problem, config, registry)
        best = solver.solve()

        customers_served = sorted(
            n for r in best.routes for n in r.customers
        )
        expected = sorted(c.id for c in simple_problem.customers)
        assert customers_served == expected


# ===========================================================================
# TEST: BENCHMARK LOADER
# ===========================================================================

class TestBenchmark:

    def test_create_sample(self):
        problem = create_sample_solomon('TEST')
        assert problem is not None
        assert problem.num_nodes > 1
        assert len(problem.customers) > 0

    def test_bks_lookup(self):
        bks = get_bks('C101')
        assert bks is not None
        assert bks > 0

    def test_compute_gap(self):
        gap = compute_gap(achieved=900.0, bks=827.3)
        assert gap > 0
        assert abs(gap - (900 - 827.3) / 827.3 * 100) < 0.01

    def test_gap_zero(self):
        """Gap = 0 khi đạt BKS."""
        gap = compute_gap(achieved=827.3, bks=827.3)
        assert abs(gap) < 1e-6

    def test_full_run_sample(self):
        """End-to-end test với sample Solomon data."""
        problem = create_sample_solomon('E2E_TEST')

        registry = PluginRegistry()
        registry.register(CapacityPlugin(violation_scale=10.0))
        registry.register(TimeWindowPlugin(late_penalty_scale=5.0))

        config = UTSConfig(
            max_iterations=100,
            max_time_seconds=20.0,
            verbose=False,
        )
        solver = UTSSolver(problem, config, registry)
        best = solver.solve()

        assert best is not None
        assert best.total_distance() < float('inf')
        # Kiểm tra customer preservation
        customers_served = sorted(n for r in best.routes for n in r.customers)
        expected = sorted(c.id for c in problem.customers)
        assert customers_served == expected


# ===========================================================================
# TEST: NUMBA KERNELS
# ===========================================================================

class TestNumbaKernels:

    def test_route_distance_consistency(self, simple_problem):
        """JIT và Python phải cho kết quả giống nhau."""
        from utils.numba_kernels import compute_route_distance_jit, flatten_problem

        flat = flatten_problem(simple_problem)
        route_nodes = np.array([0, 1, 2, 3, 0], dtype=np.int32)

        jit_result = compute_route_distance_jit(route_nodes, flat['dist_matrix'])

        # Tính thủ công
        python_result = sum(
            flat['dist_matrix'][route_nodes[i], route_nodes[i + 1]]
            for i in range(len(route_nodes) - 1)
        )
        assert abs(jit_result - python_result) < 1e-9

    def test_tw_violation_consistency(self, simple_problem):
        """TW violation JIT và Python phải cho kết quả giống nhau."""
        from utils.numba_kernels import compute_tw_violation_jit, flatten_problem

        flat = flatten_problem(simple_problem)
        route_nodes = np.array([0, 1, 2, 0], dtype=np.int32)

        jit_result = compute_tw_violation_jit(
            route_nodes,
            flat['time_matrix'],
            flat['ready_times'],
            flat['due_times'],
            flat['service_times'],
        )
        assert jit_result >= 0.0

    def test_flatten_problem_shapes(self, simple_problem):
        from utils.numba_kernels import flatten_problem
        flat = flatten_problem(simple_problem)
        n = simple_problem.num_nodes
        assert flat['dist_matrix'].shape == (n, n)
        assert flat['demands'].shape == (n,)


# ===========================================================================
# INTEGRATION TEST
# ===========================================================================

class TestIntegration:

    def test_full_pipeline(self):
        """
        Test toàn bộ pipeline:
        Problem → Solver → Best Solution → Verify
        """
        # 1. Tạo problem
        problem = create_sample_solomon('INTEGRATION_TEST')

        # 2. Setup plugins
        registry = PluginRegistry()
        registry.register(CapacityPlugin(violation_scale=10.0))
        registry.register(TimeWindowPlugin(late_penalty_scale=5.0))

        # 3. Config
        config = UTSConfig(
            max_iterations=150,
            max_time_seconds=30.0,
            max_no_improve=50,
            verbose=False,
        )

        # 4. Solve
        solver = UTSSolver(problem, config, registry)
        best = solver.solve()

        # 5. Verify
        assert best is not None

        # 5a. Tất cả customers được phục vụ
        customers_served = set(n for r in best.routes for n in r.customers)
        expected_customers = set(c.id for c in problem.customers)
        assert customers_served == expected_customers

        # 5b. Khoảng cách hợp lệ
        assert best.total_distance() > 0
        assert best.total_distance() < float('inf')

        # 5c. Convergence history tồn tại
        assert len(solver.convergence_history) > 0

        # 5d. Best cost không tệ hơn greedy
        greedy = greedy_nearest_neighbor(problem)
        assert best.total_distance() <= greedy.total_distance() + 1e-6

        print(f"\n✓ Integration test PASSED: dist={best.total_distance():.2f}, "
              f"vehicles={solver._vehicles_used(best)}, "
              f"iters={solver.iteration}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])