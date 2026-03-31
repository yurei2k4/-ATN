# benchmark/__init__.py
from benchmark.solomon_loader import load_solomon, create_sample_solomon, get_bks, compute_gap
from benchmark.runner import BenchmarkRunner, BenchmarkResult

__all__ = [
    'load_solomon',
    'create_sample_solomon',
    'get_bks',
    'compute_gap',
    'BenchmarkRunner',
    'BenchmarkResult',
]