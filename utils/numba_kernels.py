"""
utils/numba_kernels.py
----------------------
Tối ưu hóa hiệu năng bằng Numba JIT cho các hàm tính toán trọng yếu.

Selective JIT: Chỉ áp dụng Numba cho các hàm HOT PATH:
    1. Tính khoảng cách route (gọi nhiều lần nhất)
    2. Tính 2-opt delta (O(n²) mỗi iteration)
    3. Tính Relocate delta (O(n²) mỗi iteration)
    4. Kiểm tra Time Window (O(n) per route)

Nguyên lý:
    - Numba compile Python → machine code lần đầu, các lần sau gọi trực tiếp
    - Sử dụng @njit (no Python) để tối đa tốc độ
    - Cache=True để không compile lại giữa các lần chạy
    - Array-based: truyền numpy arrays thay vì objects

Fallback:
    - Nếu Numba chưa cài → dùng numpy/Python thuần
    - Kết quả hoàn toàn giống nhau, chỉ khác tốc độ
"""

from __future__ import annotations
import numpy as np
from typing import Tuple

# Thử import Numba, nếu không có thì dùng fallback
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("[Numba] OK - JIT acceleration enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("[Numba] ⚠ Numba chưa cài. Dùng numpy fallback (chậm hơn ~5-10x)")

    # Decorator giả khi không có Numba
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    def prange(*args, **kwargs):
        return range(*args, **kwargs)


# ===========================================================================
# HOT PATH 1: Tính tổng khoảng cách route
# ===========================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def compute_route_distance_jit(
        route_nodes: np.ndarray,
        dist_matrix: np.ndarray,
    ) -> float:
        """
        Tính tổng khoảng cách route với JIT acceleration.
        
        Args:
            route_nodes: Array 1D các node IDs (kể cả depot đầu/cuối)
            dist_matrix: Ma trận khoảng cách [n x n]
        
        Returns:
            Tổng khoảng cách
        """
        total = 0.0
        for i in range(len(route_nodes) - 1):
            total += dist_matrix[route_nodes[i], route_nodes[i + 1]]
        return total
else:
    def compute_route_distance_jit(
        route_nodes: np.ndarray,
        dist_matrix: np.ndarray,
    ) -> float:
        """Numpy fallback."""
        src = route_nodes[:-1]
        dst = route_nodes[1:]
        return float(dist_matrix[src, dst].sum())


# ===========================================================================
# HOT PATH 2: 2-opt Delta tất cả cặp (i, j)
# ===========================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True, parallel=True)
    def compute_two_opt_deltas_jit(
        route_nodes: np.ndarray,
        dist_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính TOÀN BỘ 2-opt deltas cho một route với parallel JIT.
        
        Returns:
            (is_arr, js_arr, deltas_arr) – chỉ các cặp có delta < 0
        """
        n = len(route_nodes)
        # Pre-allocate với kích thước tối đa
        max_pairs = n * n
        is_arr = np.empty(max_pairs, dtype=np.int32)
        js_arr = np.empty(max_pairs, dtype=np.int32)
        deltas_arr = np.empty(max_pairs, dtype=np.float64)
        count = 0

        for i in range(1, n - 2):
            a = route_nodes[i - 1]
            b = route_nodes[i]
            for j in range(i + 1, n - 1):
                c = route_nodes[j]
                d = route_nodes[j + 1]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]
                         - dist_matrix[a, b] - dist_matrix[c, d])
                if delta < -1e-9:
                    is_arr[count] = i
                    js_arr[count] = j
                    deltas_arr[count] = delta
                    count += 1

        return is_arr[:count], js_arr[:count], deltas_arr[:count]
else:
    def compute_two_opt_deltas_jit(
        route_nodes: np.ndarray,
        dist_matrix: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Numpy fallback cho 2-opt deltas."""
        n = len(route_nodes)
        is_list, js_list, deltas_list = [], [], []

        for i in range(1, n - 2):
            a, b = route_nodes[i - 1], route_nodes[i]
            for j in range(i + 1, n - 1):
                c, d = route_nodes[j], route_nodes[j + 1]
                delta = (dist_matrix[a, c] + dist_matrix[b, d]
                         - dist_matrix[a, b] - dist_matrix[c, d])
                if delta < -1e-9:
                    is_list.append(i)
                    js_list.append(j)
                    deltas_list.append(delta)

        return (np.array(is_list, dtype=np.int32),
                np.array(js_list, dtype=np.int32),
                np.array(deltas_list, dtype=np.float64))


# ===========================================================================
# HOT PATH 3: Relocate delta matrix (tất cả cặp node-route)
# ===========================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def compute_relocate_deltas_jit(
        route_a_nodes: np.ndarray,
        route_b_nodes: np.ndarray,
        dist_matrix: np.ndarray,
        demands: np.ndarray,
        capacity_b: float,
        load_b: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Tính NHANH tất cả Relocate deltas từ route_a sang route_b.
        
        Args:
            route_a_nodes: Nodes của route nguồn
            route_b_nodes: Nodes của route đích
            dist_matrix  : Ma trận khoảng cách
            demands      : Array demands[node_id]
            capacity_b   : Tải trọng tối đa của xe ở route_b
            load_b       : Tải trọng hiện tại của route_b
        
        Returns:
            (pos_from, pos_to, deltas) – chỉ các moves promising
        """
        na = len(route_a_nodes)
        nb = len(route_b_nodes)
        max_moves = na * nb
        pos_from_arr = np.empty(max_moves, dtype=np.int32)
        pos_to_arr = np.empty(max_moves, dtype=np.int32)
        delta_arr = np.empty(max_moves, dtype=np.float64)
        count = 0

        for pf in range(1, na - 1):
            u = route_a_nodes[pf]
            prev_u = route_a_nodes[pf - 1]
            next_u = route_a_nodes[pf + 1]
            demand_u = demands[u]

            # Kiểm tra capacity (short-circuit)
            if load_b + demand_u > capacity_b + 1e-6:
                continue

            removal = (dist_matrix[prev_u, next_u]
                       - dist_matrix[prev_u, u]
                       - dist_matrix[u, next_u])

            for pt in range(1, nb):
                pv = route_b_nodes[pt - 1]
                nv = route_b_nodes[pt]
                insertion = (dist_matrix[pv, u] + dist_matrix[u, nv]
                             - dist_matrix[pv, nv])
                delta = removal + insertion
                pos_from_arr[count] = pf
                pos_to_arr[count] = pt
                delta_arr[count] = delta
                count += 1

        return pos_from_arr[:count], pos_to_arr[:count], delta_arr[:count]
else:
    def compute_relocate_deltas_jit(
        route_a_nodes, route_b_nodes, dist_matrix, demands, capacity_b, load_b
    ):
        """Numpy fallback."""
        pf_list, pt_list, d_list = [], [], []
        na, nb = len(route_a_nodes), len(route_b_nodes)

        for pf in range(1, na - 1):
            u = route_a_nodes[pf]
            prev_u, next_u = route_a_nodes[pf - 1], route_a_nodes[pf + 1]
            demand_u = demands[u]
            if load_b + demand_u > capacity_b + 1e-6:
                continue
            removal = (dist_matrix[prev_u, next_u]
                       - dist_matrix[prev_u, u]
                       - dist_matrix[u, next_u])
            for pt in range(1, nb):
                pv, nv = route_b_nodes[pt - 1], route_b_nodes[pt]
                insertion = (dist_matrix[pv, u] + dist_matrix[u, nv]
                             - dist_matrix[pv, nv])
                pf_list.append(pf); pt_list.append(pt); d_list.append(removal + insertion)

        return (np.array(pf_list, dtype=np.int32),
                np.array(pt_list, dtype=np.int32),
                np.array(d_list, dtype=np.float64))


# ===========================================================================
# HOT PATH 4: Time Window violation check
# ===========================================================================

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def compute_tw_violation_jit(
        route_nodes: np.ndarray,
        time_matrix: np.ndarray,
        ready_times: np.ndarray,
        due_times: np.ndarray,
        service_times: np.ndarray,
        penalty_scale: float = 1.0,
    ) -> float:
        """
        Tính Time Window violation với JIT.
        
        Args:
            route_nodes  : Array node IDs
            time_matrix  : Ma trận thời gian di chuyển
            ready_times  : ready_times[node_id]
            due_times    : due_times[node_id]
            service_times: service_times[node_id]
            penalty_scale: Hệ số scale violation
        
        Returns:
            Tổng vi phạm (0.0 nếu feasible)
        """
        current_time = 0.0
        total_violation = 0.0
        n = len(route_nodes)

        for i in range(n - 1):
            src = route_nodes[i]
            dst = route_nodes[i + 1]
            current_time += time_matrix[src, dst]

            due = due_times[dst]
            if current_time > due + 1e-9:
                total_violation += (current_time - due) * penalty_scale

            ready = ready_times[dst]
            if current_time < ready:
                current_time = ready
            current_time += service_times[dst]

        return total_violation
else:
    def compute_tw_violation_jit(
        route_nodes, time_matrix, ready_times, due_times, service_times,
        penalty_scale=1.0
    ) -> float:
        current_time = 0.0
        total_violation = 0.0
        for i in range(len(route_nodes) - 1):
            src, dst = route_nodes[i], route_nodes[i + 1]
            current_time += time_matrix[src, dst]
            if current_time > due_times[dst] + 1e-9:
                total_violation += (current_time - due_times[dst]) * penalty_scale
            current_time = max(current_time, ready_times[dst])
            current_time += service_times[dst]
        return total_violation


# ===========================================================================
# HELPER: Chuyển đổi VRPProblem → flat arrays (Data Flattening)
# ===========================================================================

def flatten_problem(problem) -> dict:
    """
    Chuyển VRPProblem thành flat numpy arrays để dùng với JIT kernels.
    
    Returns:
        Dict với các arrays: dist_matrix, time_matrix, demands,
        ready_times, due_times, service_times
    """
    n = problem.num_nodes
    demands = np.array([problem.nodes[i].demand for i in range(n)], dtype=np.float64)
    ready_times = np.array([problem.nodes[i].ready_time for i in range(n)], dtype=np.float64)
    due_times = np.array([problem.nodes[i].due_time for i in range(n)], dtype=np.float64)
    service_times = np.array([problem.nodes[i].service_time for i in range(n)], dtype=np.float64)

    return {
        'dist_matrix': problem.dist_matrix.astype(np.float64),
        'time_matrix': problem.time_matrix.astype(np.float64),
        'demands': demands,
        'ready_times': ready_times,
        'due_times': due_times,
        'service_times': service_times,
    }


def benchmark_jit_vs_python(problem, n_trials: int = 100):
    """
    So sánh hiệu năng JIT vs Python thuần.
    In kết quả benchmark ra console.
    """
    import time as time_module

    flat = flatten_problem(problem)
    dist = flat['dist_matrix']

    # Tạo một route mẫu
    sample_route = np.array(
        [0] + list(range(1, min(20, problem.num_nodes - 1))) + [0],
        dtype=np.int32
    )

    print(f"\n{'='*50}")
    print("BENCHMARK: JIT vs Python")
    print(f"{'='*50}")
    print(f"Route length: {len(sample_route)} nodes")
    print(f"N trials: {n_trials}")
    print()

    # Warm up JIT
    if NUMBA_AVAILABLE:
        _ = compute_route_distance_jit(sample_route, dist)
        _ = compute_two_opt_deltas_jit(sample_route, dist)

    # Benchmark route distance
    t0 = time_module.perf_counter()
    for _ in range(n_trials):
        _ = compute_route_distance_jit(sample_route, dist)
    t1 = time_module.perf_counter()
    jit_time = (t1 - t0) / n_trials * 1e6  # microseconds

    # Python thuần
    def python_route_dist(nodes, dm):
        return sum(dm[nodes[i], nodes[i+1]] for i in range(len(nodes)-1))

    t0 = time_module.perf_counter()
    for _ in range(n_trials):
        _ = python_route_dist(sample_route, dist)
    t1 = time_module.perf_counter()
    python_time = (t1 - t0) / n_trials * 1e6

    print(f"Route Distance Calculation:")
    print(f"  JIT:    {jit_time:.2f} µs")
    print(f"  Python: {python_time:.2f} µs")
    if jit_time > 0:
        print(f"  Speedup: {python_time/jit_time:.1f}x")

    # Benchmark 2-opt
    t0 = time_module.perf_counter()
    for _ in range(n_trials):
        _ = compute_two_opt_deltas_jit(sample_route, dist)
    t1 = time_module.perf_counter()
    jit_2opt = (t1 - t0) / n_trials * 1000  # ms

    print(f"\n2-opt Delta Matrix ({len(sample_route)}x{len(sample_route)}):")
    print(f"  JIT: {jit_2opt:.3f} ms")
    print(f"{'='*50}\n")

    return {'jit_dist': jit_time, 'python_dist': python_time}