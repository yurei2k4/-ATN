# UTS-VRP Framework
## Ứng dụng thuật toán Unified Tabu Search trong tối ưu lộ trình giao hàng chặng cuối

### Cấu trúc dự án
```
uts_vrp/
├── core/
│   ├── __init__.py        # Public API exports
│   ├── models.py          # Data models: Node, Vehicle, Route, Solution
│   ├── solver.py          # UTS Core Engine + greedy_nearest_neighbor
│   ├── tabu_list.py       # TabuList (Zobrist Hashing) + AspirationCriteria
│   └── penalty.py         # Dynamic Penalty Controller (Strategic Oscillation)
├── plugins/
│   ├── __init__.py
│   ├── base.py            # IConstraintPlugin interface + PluginRegistry (IoC)
│   ├── capacity.py        # CapacityPlugin – ràng buộc tải trọng CVRP
│   ├── time_window.py     # TimeWindowPlugin – ràng buộc cửa sổ thời gian VRPTW
│   └── asymmetric.py      # AsymmetricRoutePlugin – đường một chiều, giờ cấm tải
├── operators/
│   ├── __init__.py
│   ├── intra_route.py     # TwoOptOperator, OrOptOperator (1/2/3-node segments)
│   └── inter_route.py     # RelocateOperator, SwapOperator, CrossExchangeOperator
├── benchmark/
│   ├── __init__.py
│   ├── solomon_loader.py  # Đọc file Solomon, BKS table 40+ instances, Gap%
│   └── runner.py          # BenchmarkRunner – chạy hàng loạt, in bảng kết quả
├── utils/
│   ├── __init__.py
│   ├── visualizer.py      # Convergence plot (matplotlib) + Route map (Folium)
│   ├── osrm_client.py     # OSRM client – ma trận khoảng cách thực tế + VietnamCityPresets
│   └── numba_kernels.py   # JIT-compiled hot-path kernels (Numba / numpy fallback)
├── tests/
│   └── test_all.py        # 35+ unit tests + integration tests (pytest)
├── data/
│   └── solomon/           # Thư mục chứa file .txt dataset Solomon
├── output/                # Thư mục lưu plots, maps, benchmark results
├── main.py                # CLI entry point (quick_test / benchmark / single)
└── requirements.txt
```

### Cài đặt
```bash
pip install -r requirements.txt
```

### Chạy benchmark Solomon
```bash
python main.py --mode benchmark --dataset data/solomon/C101.txt
```

### Chạy với dữ liệu thực tế
```bash
python main.py --mode real --city hcmc
```