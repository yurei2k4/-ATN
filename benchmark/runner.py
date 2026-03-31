"""
benchmark/runner.py
-------------------
Chạy benchmark trên dataset Solomon và tính Gap% so với BKS.

Sử dụng:
    from benchmark.runner import BenchmarkRunner
    
    runner = BenchmarkRunner(data_dir='data/solomon')
    results = runner.run(['C101', 'C102', 'R101'])
    runner.print_report(results)
"""

from __future__ import annotations
import os
import time
import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from benchmark.solomon_loader import load_solomon, get_bks, compute_gap, create_sample_solomon
from core.solver import UTSSolver, UTSConfig
from plugins.base import PluginRegistry
from plugins.capacity import CapacityPlugin
from plugins.time_window import TimeWindowPlugin


@dataclass
class BenchmarkResult:
    """Kết quả benchmark một instance."""
    instance: str
    our_distance: float
    bks: Optional[float]
    gap_percent: Optional[float]
    vehicles_used: int
    iterations: int
    time_seconds: float
    feasible: bool
    notes: str = ''


class BenchmarkRunner:
    """
    Chạy benchmark tự động trên nhiều instances Solomon.
    """

    def __init__(
        self,
        data_dir: str = 'data/solomon',
        config: UTSConfig = None,
    ):
        self.data_dir = data_dir
        self.config = config or UTSConfig(
            max_iterations=500,
            max_time_seconds=120.0,
            max_no_improve=100,
            verbose=False,
        )

    def _build_registry(self, problem_type: str) -> PluginRegistry:
        """Tạo PluginRegistry phù hợp với loại bài toán."""
        registry = PluginRegistry()
        registry.register(CapacityPlugin(violation_scale=10.0))
        if 'VRPTW' in problem_type or problem_type in ('VRPTW',):
            registry.register(TimeWindowPlugin(late_penalty_scale=5.0))
        return registry

    def run_instance(self, instance_name: str) -> BenchmarkResult:
        """
        Chạy solver trên một instance.
        
        Args:
            instance_name: Ví dụ 'C101', 'R102'
        """
        # Load problem
        filepath = os.path.join(self.data_dir, f"{instance_name}.txt")
        if os.path.exists(filepath):
            problem = load_solomon(filepath)
        else:
            print(f"  Không tìm thấy {filepath}, dùng sample data")
            problem = create_sample_solomon(instance_name)

        problem.name = instance_name
        registry = self._build_registry(problem.problem_type)

        # Chạy solver
        solver = UTSSolver(problem, self.config, registry)
        start = time.time()
        best = solver.solve()
        elapsed = time.time() - start

        # Tính kết quả
        our_dist = best.total_distance() if best else float('inf')
        bks = get_bks(instance_name)
        gap = compute_gap(our_dist, bks) if bks else None
        feasible = solver._is_feasible(best) if best else False
        vehicles = solver._vehicles_used(best) if best else 0

        return BenchmarkResult(
            instance=instance_name,
            our_distance=round(our_dist, 2),
            bks=bks,
            gap_percent=round(gap, 2) if gap is not None else None,
            vehicles_used=vehicles,
            iterations=solver.iteration,
            time_seconds=round(elapsed, 2),
            feasible=feasible,
        )

    def run(self, instances: List[str]) -> List[BenchmarkResult]:
        """
        Chạy benchmark trên danh sách instances.
        
        Args:
            instances: Danh sách tên instances ['C101', 'R101', ...]
        
        Returns:
            Danh sách BenchmarkResult
        """
        results = []
        total = len(instances)

        print(f"\n{'='*70}")
        print(f"BENCHMARK UTS-VRP FRAMEWORK")
        print(f"{'='*70}")
        print(f"{'Instance':<12} {'Dist':>10} {'BKS':>10} {'Gap%':>8} "
              f"{'Vehicles':>9} {'Time(s)':>8} {'Feasible':>9}")
        print(f"{'-'*70}")

        for i, instance in enumerate(instances, 1):
            print(f"  [{i}/{total}] {instance}...", end=' ', flush=True)
            result = self.run_instance(instance)
            results.append(result)

            gap_str = f"{result.gap_percent:.2f}%" if result.gap_percent is not None else "N/A"
            bks_str = f"{result.bks:.1f}" if result.bks else "N/A"
            feasible_str = "✓" if result.feasible else "✗"

            print(f"\r  {result.instance:<12} {result.our_distance:>10.2f} "
                  f"{bks_str:>10} {gap_str:>8} "
                  f"{result.vehicles_used:>9} {result.time_seconds:>8.2f} "
                  f"{feasible_str:>9}")

        self.print_summary(results)
        return results

    def print_report(self, results: List[BenchmarkResult]):
        """In báo cáo đầy đủ."""
        self.print_summary(results)

    def print_summary(self, results: List[BenchmarkResult]):
        """In tóm tắt kết quả."""
        print(f"\n{'='*70}")
        print("TỔNG KẾT")
        print(f"{'='*70}")

        feasible_results = [r for r in results if r.feasible]
        gap_results = [r for r in feasible_results if r.gap_percent is not None]

        print(f"  Tổng instances       : {len(results)}")
        print(f"  Feasible solutions   : {len(feasible_results)}/{len(results)}")

        if gap_results:
            avg_gap = sum(r.gap_percent for r in gap_results) / len(gap_results)
            min_gap = min(r.gap_percent for r in gap_results)
            max_gap = max(r.gap_percent for r in gap_results)
            print(f"  Average Gap%         : {avg_gap:.2f}%")
            print(f"  Best Gap%            : {min_gap:.2f}%")
            print(f"  Worst Gap%           : {max_gap:.2f}%")

        avg_time = sum(r.time_seconds for r in results) / len(results) if results else 0
        print(f"  Average time         : {avg_time:.2f}s")
        print(f"{'='*70}\n")

    def save_results(self, results: List[BenchmarkResult], filepath: str = 'benchmark_results.json'):
        """Lưu kết quả ra file JSON."""
        data = [asdict(r) for r in results]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Kết quả đã lưu: {filepath}")

    def run_quick_test(self) -> BenchmarkResult:
        """Chạy test nhanh với dữ liệu mẫu (không cần file Solomon)."""
        from benchmark.solomon_loader import create_sample_solomon
        problem = create_sample_solomon('SAMPLE_20')

        registry = PluginRegistry()
        registry.register(CapacityPlugin(violation_scale=10.0))
        registry.register(TimeWindowPlugin(late_penalty_scale=5.0))

        config = UTSConfig(
            max_iterations=200,
            max_time_seconds=30.0,
            verbose=True,
        )

        solver = UTSSolver(problem, config, registry)
        best = solver.solve()

        return BenchmarkResult(
            instance='SAMPLE_20',
            our_distance=best.total_distance() if best else float('inf'),
            bks=None,
            gap_percent=None,
            vehicles_used=solver._vehicles_used(best) if best else 0,
            iterations=solver.iteration,
            time_seconds=0.0,
            feasible=solver._is_feasible(best) if best else False,
        )