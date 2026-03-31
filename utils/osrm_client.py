"""
utils/osrm_client.py
--------------------
Lấy ma trận khoảng cách và thời gian di chuyển THỰC TẾ
từ OSRM (Open Source Routing Machine) hoặc Google Maps API.

OSRM Public Server: router.project-osrm.org (miễn phí, không cần API key)
Hỗ trợ:
    - Ma trận bất đối xứng (Asymmetric) – đường một chiều Việt Nam
    - Đơn vị: mét (khoảng cách) và giây (thời gian)
    - Batch requests để tránh rate limit

Ví dụ sử dụng:
    client = OSRMClient()
    
    locations = [
        (10.7769, 106.7009),  # Depot: Quận 1
        (10.7623, 106.6824),  # Khách hàng 1: Quận 5
        (10.8231, 106.6297),  # Khách hàng 2: Tân Bình
    ]
    
    dist_matrix, time_matrix = client.get_matrix(locations)
    problem = client.build_vrp_problem(locations, demands, vehicles)
"""

from __future__ import annotations
import time
import json
import urllib.request
import urllib.parse
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from core.models import Node, Vehicle, VRPProblem


# ===========================================================================
# OSRM CLIENT
# ===========================================================================

class OSRMClient:
    """
    Client gọi OSRM Table API để lấy ma trận khoảng cách thực tế.
    
    OSRM Table API endpoint:
        GET /table/v1/{profile}/{coordinates}
        
    Profile: 'driving' (xe tải), 'cycling', 'foot'
    
    Rate limiting: Public server có giới hạn. Nên dùng OSRM self-hosted
    hoặc chia nhỏ requests (batch_size).
    """

    OSRM_PUBLIC = "https://router.project-osrm.org"
    OSRM_DEMO = "https://routing.openstreetmap.de/routed-car"

    def __init__(
        self,
        base_url: str = None,
        profile: str = 'driving',
        batch_size: int = 100,
        timeout: int = 30,
        retry: int = 3,
    ):
        """
        Args:
            base_url  : URL OSRM server (None = dùng public server)
            profile   : Loại phương tiện ('driving', 'cycling', 'foot')
            batch_size: Số điểm tối đa mỗi request (public server giới hạn ~100)
            timeout   : Timeout HTTP request (giây)
            retry     : Số lần thử lại khi thất bại
        """
        self.base_url = base_url or self.OSRM_PUBLIC
        self.profile = profile
        self.batch_size = batch_size
        self.timeout = timeout
        self.retry = retry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_matrix(
        self,
        locations: List[Tuple[float, float]],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy ma trận khoảng cách và thời gian di chuyển.
        
        Args:
            locations: List of (latitude, longitude) tuples
                       VÍ DỤ: [(10.7769, 106.7009), (10.7623, 106.6824)]
        
        Returns:
            (dist_matrix, time_matrix) – numpy arrays [n x n]
            dist_matrix: khoảng cách thực tế (mét)
            time_matrix: thời gian di chuyển (giây)
        
        Notes:
            - Ma trận có thể bất đối xứng nếu có đường một chiều
            - Giá trị trên đường chéo = 0
        """
        n = len(locations)

        if verbose:
            print(f"Lấy ma trận {n}x{n} từ OSRM ({self.base_url})")
            print(f"Profile: {self.profile}")

        if n <= self.batch_size:
            # Single request
            return self._fetch_matrix_single(locations, verbose)
        else:
            # Batch requests
            return self._fetch_matrix_batched(locations, verbose)

    def get_route(
        self,
        waypoints: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """
        Lấy lộ trình cụ thể giữa các waypoints.
        
        Returns:
            Dict với keys: distance (m), duration (s), geometry (polyline)
        """
        coords_str = ';'.join(f"{lon},{lat}" for lat, lon in waypoints)
        url = (f"{self.base_url}/route/v1/{self.profile}/{coords_str}"
               f"?overview=simplified&geometries=geojson")

        data = self._request(url)
        if not data or 'routes' not in data:
            return {}

        route = data['routes'][0]
        return {
            'distance': route.get('distance', 0),
            'duration': route.get('duration', 0),
            'geometry': route.get('geometry', {}),
        }

    def build_vrp_problem(
        self,
        locations: List[Tuple[float, float]],
        demands: List[float],
        vehicles: List[Vehicle],
        time_windows: Optional[List[Tuple[float, float]]] = None,
        service_times: Optional[List[float]] = None,
        problem_name: str = 'real_world',
        dist_unit: str = 'km',
        time_unit: str = 'minutes',
    ) -> VRPProblem:
        """
        Tạo VRPProblem từ tọa độ thực tế.
        
        Args:
            locations   : [(lat, lon), ...] – index 0 = depot
            demands     : [0, d1, d2, ...] – demand tại mỗi điểm (0 = depot)
            vehicles    : Danh sách xe
            time_windows: [(ready, due), ...] – time window (None = không giới hạn)
            service_times: [0, s1, s2, ...] – thời gian phục vụ tại mỗi điểm
            dist_unit   : 'km' hoặc 'm' (đơn vị khoảng cách đầu ra)
            time_unit   : 'minutes' hoặc 'seconds' (đơn vị thời gian đầu ra)
        
        Returns:
            VRPProblem sẵn sàng để solve
        """
        n = len(locations)
        assert len(demands) == n, "demands phải có cùng độ dài với locations"

        # Lấy ma trận từ OSRM
        dist_m, time_s = self.get_matrix(locations)

        # Chuyển đơn vị
        if dist_unit == 'km':
            dist_matrix = dist_m / 1000.0
        else:
            dist_matrix = dist_m

        if time_unit == 'minutes':
            time_matrix = time_s / 60.0
        else:
            time_matrix = time_s

        # Tạo nodes
        nodes = []
        for i, (lat, lon) in enumerate(locations):
            tw = time_windows[i] if time_windows else (0.0, float('inf'))
            st = service_times[i] if service_times else 0.0

            nodes.append(Node(
                id=i,
                x=lon,   # x = longitude
                y=lat,   # y = latitude
                demand=demands[i],
                ready_time=tw[0],
                due_time=tw[1],
                service_time=st,
                node_type='depot' if i == 0 else 'customer',
            ))

        problem_type = 'VRPTW' if time_windows else 'AVRP'

        return VRPProblem(
            nodes=nodes,
            vehicles=vehicles,
            dist_matrix=dist_matrix,
            time_matrix=time_matrix,
            problem_type=problem_type,
            name=problem_name,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _fetch_matrix_single(
        self,
        locations: List[Tuple[float, float]],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lấy ma trận trong một request duy nhất."""
        n = len(locations)

        # Format: lon,lat;lon,lat;...
        coords_str = ';'.join(f"{lon},{lat}" for lat, lon in locations)
        url = (f"{self.base_url}/table/v1/{self.profile}/{coords_str}"
               f"?annotations=duration,distance")

        if verbose:
            print(f"  Request: {n} locations...")

        data = self._request(url)

        if not data:
            if verbose:
                print("  OSRM không khả dụng, dùng Euclidean fallback")
            return self._euclidean_fallback(locations)

        # Parse response
        dist_matrix = np.array(data.get('distances', []), dtype=float)
        time_matrix = np.array(data.get('durations', []), dtype=float)

        # Thay thế None bằng giá trị lớn (disconnected nodes)
        dist_matrix = np.where(dist_matrix is None, 1e9, dist_matrix)
        time_matrix = np.where(time_matrix is None, 1e9, time_matrix)

        if verbose:
            print(f"  ✓ Nhận được ma trận {dist_matrix.shape}")
            is_asymmetric = not np.allclose(dist_matrix, dist_matrix.T, atol=1.0)
            print(f"  Bất đối xứng: {'CÓ (đường một chiều phát hiện)' if is_asymmetric else 'KHÔNG'}")

        return dist_matrix, time_matrix

    def _fetch_matrix_batched(
        self,
        locations: List[Tuple[float, float]],
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Lấy ma trận cho nhiều điểm bằng cách chia batch.
        
        Chiến lược: OSRM Table API cho phép chỉ định sources và destinations riêng.
        Chia thành nhiều requests, mỗi request lấy một phần của ma trận.
        """
        n = len(locations)
        dist_matrix = np.zeros((n, n))
        time_matrix = np.zeros((n, n))

        # Tất cả locations làm coordinates
        coords_str = ';'.join(f"{lon},{lat}" for lat, lon in locations)

        # Chia sources thành batches
        batch_size = self.batch_size
        n_batches = (n + batch_size - 1) // batch_size

        if verbose:
            print(f"  Chia thành {n_batches} batches (batch_size={batch_size})")

        for batch_idx in range(n_batches):
            src_start = batch_idx * batch_size
            src_end = min(src_start + batch_size, n)
            sources = list(range(src_start, src_end))
            destinations = list(range(n))

            src_str = ';'.join(map(str, sources))
            dst_str = ';'.join(map(str, destinations))

            url = (f"{self.base_url}/table/v1/{self.profile}/{coords_str}"
                   f"?sources={src_str}&destinations={dst_str}"
                   f"&annotations=duration,distance")

            data = self._request(url)
            if data:
                batch_dist = np.array(data.get('distances', []), dtype=float)
                batch_time = np.array(data.get('durations', []), dtype=float)
                dist_matrix[src_start:src_end, :] = batch_dist
                time_matrix[src_start:src_end, :] = batch_time
            else:
                # Fallback cho batch này
                for i in sources:
                    for j in destinations:
                        d = self._haversine(locations[i], locations[j])
                        dist_matrix[i, j] = d * 1000  # km → m
                        time_matrix[i, j] = d / 40 * 3600  # Giả sử 40 km/h

            if verbose:
                print(f"  Batch {batch_idx + 1}/{n_batches} ✓")

            # Rate limiting
            time.sleep(0.5)

        return dist_matrix, time_matrix

    def _request(self, url: str) -> Optional[dict]:
        """Gửi HTTP request với retry logic."""
        for attempt in range(self.retry):
            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'UTS-VRP-Framework/1.0'}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode('utf-8'))
                    if data.get('code') == 'Ok':
                        return data
                    else:
                        print(f"  OSRM error: {data.get('message', 'Unknown')}")
                        return None
            except Exception as e:
                if attempt < self.retry - 1:
                    print(f"  Retry {attempt + 1}/{self.retry}: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"  Request thất bại sau {self.retry} lần: {e}")
        return None

    def _euclidean_fallback(
        self,
        locations: List[Tuple[float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback khi OSRM không khả dụng:
        Dùng khoảng cách Haversine (đường chim bay) nhân với hệ số đường bộ.
        """
        n = len(locations)
        dist_matrix = np.zeros((n, n))
        ROAD_FACTOR = 1.3   # Hệ số đường bộ ~30% dài hơn đường chim bay
        AVG_SPEED = 30.0    # km/h (tốc độ trung bình đô thị Việt Nam)

        for i in range(n):
            for j in range(n):
                if i != j:
                    d_km = self._haversine(locations[i], locations[j])
                    dist_matrix[i, j] = d_km * ROAD_FACTOR * 1000  # → mét

        time_matrix = dist_matrix / (AVG_SPEED * 1000 / 3600)  # mét/giây → giây
        return dist_matrix, time_matrix

    @staticmethod
    def _haversine(loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        """
        Tính khoảng cách Haversine giữa 2 điểm (km).
        Chính xác hơn Euclidean cho tọa độ địa lý.
        """
        lat1, lon1 = loc1
        lat2, lon2 = loc2
        R = 6371.0  # Bán kính Trái Đất (km)

        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)

        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ===========================================================================
# PRESET LOCATIONS – Dữ liệu mẫu các đô thị Việt Nam
# ===========================================================================

class VietnamCityPresets:
    """
    Tập hợp tọa độ mẫu cho các khu vực đô thị lớn Việt Nam.
    Dùng để test nhanh với dữ liệu thực tế.
    """

    HCMC_DISTRICTS = {
        'depot_q1': (10.7769, 106.7009),      # Quận 1 – Depot trung tâm
        'q2_thao_dien': (10.8013, 106.7430),  # Quận 2
        'q3_vo_thi_sau': (10.7731, 106.6903), # Quận 3
        'q5_cho_lon': (10.7484, 106.6625),    # Quận 5
        'q7_phu_my_hung': (10.7291, 106.7206),# Quận 7
        'q9_long_truong': (10.8249, 106.8006),# Quận 9
        'q10_nguyen_trai': (10.7735, 106.6640),# Quận 10
        'binh_thanh': (10.8080, 106.7130),    # Bình Thạnh
        'tan_binh': (10.8051, 106.6523),      # Tân Bình
        'go_vap': (10.8413, 106.6716),        # Gò Vấp
        'binh_duong': (10.9804, 106.6519),    # Bình Dương (ngoại thành)
    }

    HANOI_DISTRICTS = {
        'depot_hoan_kiem': (21.0285, 105.8542), # Hoàn Kiếm – Depot
        'dong_da': (21.0218, 105.8412),          # Đống Đa
        'hai_ba_trung': (21.0036, 105.8600),     # Hai Bà Trưng
        'ba_dinh': (21.0372, 105.8209),          # Ba Đình
        'tay_ho': (21.0693, 105.8225),           # Tây Hồ
        'cau_giay': (21.0359, 105.7939),         # Cầu Giấy
        'thanh_xuan': (20.9964, 105.8091),       # Thanh Xuân
        'hoang_mai': (20.9802, 105.8498),        # Hoàng Mai
        'long_bien': (21.0447, 105.8887),        # Long Biên
        'ha_dong': (20.9627, 105.7815),          # Hà Đông
    }

    @classmethod
    def get_hcmc_sample(
        cls,
        n_customers: int = 10,
        n_vehicles: int = 3,
        capacity: float = 50.0,
    ) -> Tuple[List[Tuple[float, float]], List[float], List['Vehicle']]:
        """
        Trả về sample data cho TP.HCM.
        
        Returns:
            (locations, demands, vehicles)
        """
        from core.models import Vehicle
        import random
        random.seed(42)

        all_locations = list(cls.HCMC_DISTRICTS.values())
        depot = all_locations[0]
        customers = all_locations[1:n_customers + 1]

        locations = [depot] + customers
        demands = [0.0] + [random.uniform(5, 20) for _ in customers]
        vehicles = [Vehicle(id=i, capacity=capacity) for i in range(n_vehicles)]

        return locations, demands, vehicles

    @classmethod
    def get_hanoi_sample(
        cls,
        n_customers: int = 9,
        n_vehicles: int = 3,
        capacity: float = 50.0,
    ) -> Tuple[List[Tuple[float, float]], List[float], List['Vehicle']]:
        """Trả về sample data cho Hà Nội."""
        from core.models import Vehicle
        import random
        random.seed(42)

        all_locations = list(cls.HANOI_DISTRICTS.values())
        depot = all_locations[0]
        customers = all_locations[1:n_customers + 1]

        locations = [depot] + customers
        demands = [0.0] + [random.uniform(5, 20) for _ in customers]
        vehicles = [Vehicle(id=i, capacity=capacity) for i in range(n_vehicles)]

        return locations, demands, vehicles


def build_real_world_problem(
    city: str = 'hcmc',
    use_osrm: bool = True,
) -> 'VRPProblem':
    """
    Convenience function: tạo VRPProblem từ dữ liệu thực tế.
    
    Args:
        city    : 'hcmc' hoặc 'hanoi'
        use_osrm: True = lấy khoảng cách thực từ OSRM,
                  False = dùng Haversine fallback
    
    Returns:
        VRPProblem với ma trận khoảng cách thực tế
    """
    presets = VietnamCityPresets()

    if city.lower() in ('hcmc', 'hồ_chí_minh', 'saigon'):
        locations, demands, vehicles = presets.get_hcmc_sample()
        name = 'HCMC_Real'
    else:
        locations, demands, vehicles = presets.get_hanoi_sample()
        name = 'Hanoi_Real'

    client = OSRMClient()

    if use_osrm:
        return client.build_vrp_problem(
            locations=locations,
            demands=demands,
            vehicles=vehicles,
            problem_name=name,
        )
    else:
        # Dùng Haversine fallback (không cần internet)
        dist_m, time_s = client._euclidean_fallback(locations)
        from core.models import Node, VRPProblem
        nodes = [
            Node(
                id=i, x=lon, y=lat,
                demand=demands[i],
                node_type='depot' if i == 0 else 'customer'
            )
            for i, (lat, lon) in enumerate(locations)
        ]
        return VRPProblem(
            nodes=nodes,
            vehicles=vehicles,
            dist_matrix=dist_m / 1000.0,  # → km
            time_matrix=time_s / 60.0,    # → phút
            problem_type='AVRP',
            name=name + '_Haversine',
        )