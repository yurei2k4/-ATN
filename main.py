"""
main.py
-------
Entry point chính của framework UTS-VRP.

Các chế độ chạy:
    1. quick_test  : Test nhanh với dữ liệu mẫu (không cần file)
    2. benchmark   : Benchmark trên dataset Solomon
    3. single      : Giải một instance cụ thể
    
Ví dụ:
    python main.py                              # quick test
    python main.py --mode benchmark             # benchmark tất cả
    python main.py --mode single --file C101    # giải C101
    python main.py --mode single --file C101 --iterations 2000
"""

import argparse
import os
import sys
import logging

# Thêm thư mục gốc vào Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.solver import UTSSolver, UTSConfig
from plugins.base import PluginRegistry
from plugins.capacity import CapacityPlugin
from plugins.time_window import TimeWindowPlugin
from plugins.asymmetric import AsymmetricRoutePlugin
from benchmark.solomon_loader import (
    load_solomon, create_sample_solomon, get_bks, compute_gap
)
from benchmark.runner import BenchmarkRunner
from utils.visualizer import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)


# ===========================================================================
# QUICK TEST
# ===========================================================================

def run_quick_test(args):
    """
    Chạy nhanh với 20 khách hàng mẫu.
    Không cần file dữ liệu.
    """
    print("\n" + "="*60)
    print("QUICK TEST – 20 Customers VRPTW Sample")
    print("="*60)

    problem = create_sample_solomon('VRPTW_SAMPLE_20')

    # Setup plugins
    registry = PluginRegistry()
    registry.register(CapacityPlugin(violation_scale=10.0))
    registry.register(TimeWindowPlugin(late_penalty_scale=5.0))

    config = UTSConfig(
        max_iterations=args.iterations,
        max_time_seconds=args.time_limit,
        max_no_improve=150,
        verbose=True,
        log_interval=50,
    )

    solver = UTSSolver(problem, config, registry)
    best = solver.solve()

    if best:
        stats = solver.get_stats()
        print(f"\nKết quả cuối cùng:")
        print(f"  Best distance  : {stats['best_distance']:.2f}")
        print(f"  Vehicles used  : {stats['vehicles_used']}")
        print(f"  Iterations     : {stats['iterations']}")
        print(f"  Time           : {stats['total_time']:.2f}s")
        print(f"  Feasible       : {solver._is_feasible(best)}")

        # Visualize
        viz = Visualizer(problem, best, solver)
        viz.print_solution_table()

        if args.save_plots:
            os.makedirs('output', exist_ok=True)
            viz.plot_routes_matplotlib('output/routes_sample.png')
            viz.plot_convergence('output/convergence_sample.png')
            print("\nPlots đã lưu vào thư mục output/")

    return best, solver


# ===========================================================================
# BENCHMARK MODE
# ===========================================================================

def run_benchmark(args):
    """
    Chạy benchmark trên Solomon dataset.
    """
    config = UTSConfig(
        max_iterations=args.iterations,
        max_time_seconds=args.time_limit,
        max_no_improve=150,
        verbose=False,
    )

    runner = BenchmarkRunner(
        data_dir=args.data_dir,
        config=config,
    )

    # Danh sách instances mặc định
    if args.instances:
        instances = args.instances.split(',')
    else:
        instances = ['C101', 'C102', 'R101', 'R102', 'RC101', 'RC102']

    results = runner.run(instances)

    if args.save_results:
        os.makedirs('output', exist_ok=True)
        runner.save_results(results, 'output/benchmark_results.json')

    return results


# ===========================================================================
# SINGLE INSTANCE MODE
# ===========================================================================

def run_single(args):
    """
    Giải một instance cụ thể với đầy đủ visualization.
    """
    # Load problem
    if args.file:
        filepath = os.path.join(args.data_dir, f"{args.file}.txt")
        if os.path.exists(filepath):
            problem = load_solomon(filepath)
            problem.name = args.file
        else:
            print(f"Không tìm thấy {filepath}, dùng sample data")
            problem = create_sample_solomon(args.file)
    else:
        problem = create_sample_solomon('DEFAULT_SAMPLE')

    print(f"\nBài toán: {problem.name}")
    print(f"  Loại: {problem.problem_type}")
    print(f"  Nodes: {problem.num_nodes} ({len(problem.customers)} customers)")
    print(f"  Vehicles: {problem.num_vehicles}")

    # Setup plugins
    registry = PluginRegistry()
    registry.register(CapacityPlugin(violation_scale=10.0))

    if problem.problem_type == 'VRPTW':
        registry.register(TimeWindowPlugin(late_penalty_scale=5.0))
        print("  Plugins: CapacityPlugin + TimeWindowPlugin")
    else:
        print("  Plugins: CapacityPlugin")

    # Tùy chọn thêm AsymmetricPlugin
    if args.asymmetric:
        registry.register(AsymmetricRoutePlugin())
        print("  + AsymmetricRoutePlugin")

    config = UTSConfig(
        max_iterations=args.iterations,
        max_time_seconds=args.time_limit,
        max_no_improve=200,
        verbose=True,
        log_interval=50,
    )

    # Solve
    solver = UTSSolver(problem, config, registry)
    best = solver.solve()

    if best:
        # In kết quả
        viz = Visualizer(problem, best, solver)
        viz.print_solution_table()

        stats = solver.get_stats()
        bks = get_bks(problem.name)
        if bks:
            gap = compute_gap(stats['best_distance'], bks)
            print(f"  BKS: {bks:.2f} | Gap: {gap:.2f}%")

        # Lưu plots
        if args.save_plots:
            os.makedirs('output', exist_ok=True)
            name_clean = problem.name.replace(' ', '_')
            viz.plot_routes_matplotlib(f'output/routes_{name_clean}.png')
            viz.plot_convergence(f'output/convergence_{name_clean}.png')
            viz.plot_routes_folium(f'output/routes_{name_clean}.html')
            print(f"\nĐã lưu output vào thư mục output/")

    return best, solver


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='UTS-VRP Framework – Unified Tabu Search cho Last-Mile Delivery'
    )

    parser.add_argument(
        '--mode', choices=['quick_test', 'benchmark', 'single'],
        default='quick_test',
        help='Chế độ chạy'
    )
    parser.add_argument('--file', type=str, default=None,
                        help='Tên instance Solomon (ví dụ: C101)')
    parser.add_argument('--data-dir', type=str, default='data/solomon',
                        help='Thư mục chứa dữ liệu Solomon')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Số iterations tối đa')
    parser.add_argument('--time-limit', type=float, default=60.0,
                        help='Giới hạn thời gian (giây)')
    parser.add_argument('--instances', type=str, default=None,
                        help='Danh sách instances benchmark, phân cách bởi dấu phẩy')
    parser.add_argument('--save-plots', action='store_true',
                        help='Lưu đồ thị hội tụ và bản đồ lộ trình')
    parser.add_argument('--save-results', action='store_true',
                        help='Lưu kết quả benchmark ra JSON')
    parser.add_argument('--asymmetric', action='store_true',
                        help='Bật AsymmetricRoutePlugin')

    args = parser.parse_args()

    if args.mode == 'quick_test':
        run_quick_test(args)
    elif args.mode == 'benchmark':
        run_benchmark(args)
    elif args.mode == 'single':
        run_single(args)


if __name__ == '__main__':
    main()