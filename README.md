# UTS-VRP Framework 🚚

**Dự án: Ứng dụng thuật toán Unified Tabu Search (UTS) trong tối ưu lộ trình giao hàng chặng cuối (Vehicle Routing Problem).**

Framework hỗ trợ tính toán giải quyết các bài toán VRP đa dạng:
- **CVRP** (Capacitated VRP): Giới hạn tải trọng xe.
- **VRPTW** (VRP with Time Windows): Cửa sổ thời gian phục vụ tại mỗi điểm.
- **Asymmetric VRP**: Khoảng cách đường đi 2 chiều khác nhau (đường một chiều, đường đô thị thực tế).

Được tối ưu hóa hiệu năng cực cao bằng **Numba JIT** và có khả năng tích hợp dữ liệu khoảng cách xe thực tế từ **OSRM**.

---

## 🛠 Cài đặt

Một môi trường Python ảo (`venv`) được khuyến khích sử dụng. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```
*Các dependency chính bao gồm: `numpy`, `numba` (tăng tốc độ chạy 5-10x), `matplotlib`, `folium` (vẽ bản đồ), `pytest` (kiểm thử).*

---

## 📦 Bộ dữ liệu (Dataset)

Dự án hỗ trợ benchmark trên bộ dữ liệu học thuật **Solomon's VRPTW Benchmark**, nhưng mặc định kho lưu trữ không đi kèm file dữ liệu này. 

**Vấn đề Tải tự động (HTTP 404):** Link tĩnh của server tải tự động Solomon tại `sintef.no` hiện đã hỏng, do vậy bạn cần tải file dữ liệu thủ công nếu muốn chạy benchmark đầy đủ:

1. Tải bộ dữ liệu Solomon (.txt) chuẩn từ trang chủ: [http://web.cba.neu.edu/~msolomon/problems.htm](http://web.cba.neu.edu/~msolomon/problems.htm)
2. Giải nén tất cả các file cấu hình như `C101.txt`, `R102.txt`... vào thư mục `data/solomon/`.
3. *(Lưu ý: Không bắt buộc phải có Solomon để chạy `quick_test` vì framework có khả năng tự sinh dữ liệu ngẫu nhiên).*

---

## 🚀 Hướng dẫn Sử dụng (CLI)

Framework sử dụng `main.py` làm entry point tương tác.

### 1. Quick Test (Dữ liệu sinh ngẫu nhiên)
Chạy thử nghiệm nhanh với dữ liệu synthetic (20 customers) mà không cần mạng hoặc file cài đặt. Phù hợp để kiểm tra code:
```bash
python main.py --mode quick_test --iterations 50
```

### 2. Dữ liệu thực tế TP.HCM & Hà Nội (API OSRM)
Tính toán tuyến đường giao hàng ở môi trường thực tế với tọa độ GPS, ma trận khoảng cách chuẩn lấy từ `router.project-osrm.org`. Sau khi chạy, kết quả sẽ vẽ ra một bản đồ tương tác đẹp mắt (HTML):
```bash
python main.py --mode real --city hcmc
``` 
*(City có thể đổi thành `hanoi`). Kết quả bản đồ tự động lưu tại `output/routes_map.html` để có thể xem trên trình duyệt.*

### 3. Đánh giá Benchmark (Chạy trên file .txt)
Sau khi đã thêm các file dữ liệu vào thư mục `data/solomon/`:
```bash
python main.py --mode benchmark
```

---

## 🗺️ Bản đồ Folium (Routes Map)

Chức năng **trực quan hóa bằng Folium** đã được cải tiến để render tuyến đường rất chi tiết dành riêng cho dữ liệu thực với các tính năng:
- **Tự động lấy tâm bản đồ (Auto Center)** dựa trên trung bình các điểm.
- **Richer popups & Tooltips**: Hiển thị Demand, Arrival Time, Time Window, và thứ tự điểm dừng của xe.
- **Vẽ đường:** Nhúng polyline từng xe, scale màu tự động.
- *(Nếu cố tình dùng Folium render dữ liệu synthetic Solomon Euclidean, hệ thống sẽ tự động phát hiện, cảnh báo, và scale giả lập về vùng bản đồ TP.HCM để không bị lỗi trống bản đồ).*

---

## 🧪 Kiểm thử (Testing)

Dự án có độ phủ test toàn diện (35+ test cases) chứng minh độ bền bỉ của Tabu Search và mô hình Constraint Plugins.
```bash
python -m pytest tests/test_all.py -v --tb=short
```

---

*Project Maintainer: [@yurei2k4](https://github.com/yurei2k4/-ATN)*