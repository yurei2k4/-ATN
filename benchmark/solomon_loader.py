"""
benchmark/solomon_loader.py
---------------------------
Đọc dữ liệu chuẩn Solomon cho bài toán VRPTW.

Format file Solomon:
    Line 1: Tên bài toán
    Line 4: Số xe tối đa, Tải trọng mỗi xe
    Line 9+: CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME
    
    Node 0 = Depot
    Node 1..n = Customers

Dataset Solomon có 6 nhóm:
    C1, C2: Clustered customers, tight/loose TW
    R1, R2: Random customers, tight/loose TW
    RC1, RC2: Random-clustered, tight/loose TW

Best Known Solutions (BKS) cho subset phổ biến được lưu trong BKS_TABLE.
"""

from __future__ import annotations
import os
import re
from typing import Optional, Dict, List, Tuple
import numpy as np

from core.models import Node, Vehicle, VRPProblem


# ---------------------------------------------------------------------------
# Best Known Solutions (BKS) – dùng để tính Gap%
# Nguồn: http://www.vrp-rep.org / Gehring & Homberger, 1999
# ---------------------------------------------------------------------------
BKS_TABLE: Dict[str, float] = {
    # C1 group (25 customers per route, tight TW)
    'C101': 827.3,
    'C102': 827.3,
    'C103': 826.3,
    'C104': 822.9,
    'C105': 827.3,
    'C106': 827.3,
    'C107': 827.3,
    'C108': 827.3,
    'C109': 827.3,
    # C2 group
    'C201': 589.1,
    'C202': 589.1,
    'C203': 588.7,
    'C204': 588.1,
    'C205': 586.4,
    'C206': 586.0,
    'C207': 585.8,
    'C208': 585.8,
    # R1 group
    'R101': 1637.7,
    'R102': 1466.6,
    'R103': 1208.7,
    'R104': 971.5,
    'R105': 1355.3,
    'R106': 1234.6,
    'R107': 1064.6,
    'R108': 932.1,
    'R109': 1146.9,
    'R110': 1068.0,
    'R111': 1048.7,
    'R112': 948.6,
    # R2 group
    'R201': 1143.2,
    'R202': 1029.6,
    'R203': 870.8,
    'R204': 731.3,
    'R205': 949.8,
    'R206': 875.9,
    'R207': 794.0,
    'R208': 701.0,
    # RC1 group
    'RC101': 1619.8,
    'RC102': 1457.4,
    'RC103': 1258.0,
    'RC104': 1132.3,
    'RC105': 1513.7,
    'RC106': 1372.7,
    'RC107': 1207.8,
    'RC108': 1114.2,
    # RC2 group
    'RC201': 1261.8,
    'RC202': 1092.3,
    'RC203': 923.7,
    'RC204': 783.5,
    'RC205': 1154.0,
    'RC206': 1051.1,
    'RC207': 962.9,
    'RC208': 776.1,
}


def load_solomon(filepath: str) -> VRPProblem:
    """
    Đọc file Solomon và tạo VRPProblem.
    
    Args:
        filepath: Đường dẫn đến file .txt Solomon
    
    Returns:
        VRPProblem đã được khởi tạo
    
    Example:
        problem = load_solomon('data/solomon/C101.txt')
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Tên bài toán (line 1)
    name = lines[0].strip()

    # Số xe và tải trọng (line 4, index 3)
    vehicle_line = lines[3].split() if len(lines) > 3 else lines[4].split()
    # Tìm dòng có "NUMBER" → dòng tiếp theo là số liệu
    for i, line in enumerate(lines):
        if 'NUMBER' in line.upper() and 'CAPACITY' in line.upper():
            parts = lines[i + 1].split()
            num_vehicles = int(parts[0])
            capacity = float(parts[1])
            break
    else:
        # Fallback
        num_vehicles = 25
        capacity = 200.0

    # Tìm dòng bắt đầu dữ liệu nodes
    data_start = None
    for i, line in enumerate(lines):
        if re.match(r'\s*0\s+', line) and len(line.split()) == 7:
            data_start = i
            break
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and len(stripped.split()) == 7:
            try:
                int(stripped.split()[0])
                data_start = i
                break
            except ValueError:
                continue

    if data_start is None:
        raise ValueError(f"Không tìm thấy dữ liệu node trong file {filepath}")

    # Đọc nodes
    nodes = []
    for line in lines[data_start:]:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            node_id = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            demand = float(parts[3])
            ready_time = float(parts[4])
            due_time = float(parts[5])
            service_time = float(parts[6])

            node_type = 'depot' if node_id == 0 else 'customer'
            nodes.append(Node(
                id=node_id,
                x=x, y=y,
                demand=demand,
                ready_time=ready_time,
                due_time=due_time,
                service_time=service_time,
                node_type=node_type,
            ))
        except (ValueError, IndexError):
            continue

    if not nodes:
        raise ValueError(f"Không đọc được node nào từ {filepath}")

    # Re-index nodes (đảm bảo 0-indexed và liên tục)
    nodes.sort(key=lambda n: n.id)
    for i, node in enumerate(nodes):
        node.id = i

    # Tạo vehicles
    vehicles = [Vehicle(id=i, capacity=capacity) for i in range(num_vehicles)]

    # Tính Euclidean distance matrix
    n = len(nodes)
    coords = np.array([[node.x, node.y] for node in nodes])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    return VRPProblem(
        nodes=nodes,
        vehicles=vehicles,
        dist_matrix=dist_matrix,
        problem_type='VRPTW',
        name=name,
    )


def get_bks(problem_name: str) -> Optional[float]:
    """
    Lấy Best Known Solution cho một bài toán Solomon.
    
    Args:
        problem_name: Tên bài toán (ví dụ: 'C101', 'R102')
    """
    key = problem_name.upper().replace('.TXT', '').strip()
    return BKS_TABLE.get(key)


def compute_gap(achieved: float, bks: float) -> float:
    """
    Tính Gap% so với BKS:
        Gap% = (achieved - BKS) / BKS * 100
    
    Gap% = 0 → đạt BKS
    Gap% < 5% → kết quả rất tốt
    Gap% < 10% → kết quả tốt
    """
    if bks <= 0:
        return float('inf')
    return (achieved - bks) / bks * 100.0


def download_solomon_data(output_dir: str = 'data/solomon'):
    """
    Tải dữ liệu Solomon từ URL công khai.
    (Chạy một lần để setup data)
    """
    import urllib.request
    os.makedirs(output_dir, exist_ok=True)

    # URL công khai của dataset Solomon
    BASE_URL = "https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/download/"

    instances = [
        'C101', 'C102', 'C201',
        'R101', 'R102', 'R201',
        'RC101', 'RC102', 'RC201',
    ]

    print(f"Tải Solomon dataset vào {output_dir}/")
    for name in instances:
        url = f"{BASE_URL}{name}.txt"
        dst = os.path.join(output_dir, f"{name}.txt")
        if os.path.exists(dst):
            print(f"  {name}.txt đã tồn tại, skip")
            continue
        try:
            urllib.request.urlretrieve(url, dst)
            print(f"  ✓ {name}.txt")
        except Exception as e:
            print(f"  ✗ {name}.txt: {e}")


def create_sample_solomon(name: str = 'C101_SAMPLE') -> VRPProblem:
    """
    Tạo một bài toán Solomon mẫu nhỏ để test (không cần file).
    20 customers, 5 xe, capacity=100.
    """
    np.random.seed(42)
    n_customers = 20
    n_vehicles = 5
    capacity = 100.0

    # Depot ở trung tâm
    depot = Node(
        id=0, x=50.0, y=50.0,
        demand=0.0, ready_time=0.0, due_time=1000.0,
        service_time=0.0, node_type='depot'
    )

    # Customers ngẫu nhiên
    nodes = [depot]
    for i in range(1, n_customers + 1):
        nodes.append(Node(
            id=i,
            x=float(np.random.uniform(0, 100)),
            y=float(np.random.uniform(0, 100)),
            demand=float(np.random.randint(5, 25)),
            ready_time=float(np.random.uniform(0, 400)),
            due_time=float(np.random.uniform(500, 1000)),
            service_time=10.0,
            node_type='customer'
        ))

    vehicles = [Vehicle(id=i, capacity=capacity) for i in range(n_vehicles)]

    coords = np.array([[n.x, n.y] for n in nodes])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    return VRPProblem(
        nodes=nodes,
        vehicles=vehicles,
        dist_matrix=dist_matrix,
        problem_type='VRPTW',
        name=name,
    )