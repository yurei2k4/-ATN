"""
utils/visualizer.py
-------------------
Trực quan hóa kết quả:
    1. Đồ thị hội tụ (Convergence Plot): chi phí và λ theo iteration
    2. Bản đồ lộ trình (Route Map): visualize routes trên Folium map

Sử dụng:
    viz = Visualizer(problem, solution, solver)
    viz.plot_convergence('convergence.png')
    viz.plot_routes_folium('routes.html')
    viz.plot_routes_matplotlib('routes.png')
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, List, Optional, Dict, Any
import numpy as np

if TYPE_CHECKING:
    from core.models import VRPProblem, Solution
    from core.solver import UTSSolver


# Màu sắc cho các routes
ROUTE_COLORS = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
]


class Visualizer:
    """
    Lớp tổng hợp các chức năng visualization.
    """

    def __init__(
        self,
        problem: 'VRPProblem',
        solution: 'Solution',
        solver: Optional['UTSSolver'] = None,
    ):
        self.problem = problem
        self.solution = solution
        self.solver = solver

    # ------------------------------------------------------------------
    # 1. CONVERGENCE PLOT
    # ------------------------------------------------------------------

    def plot_convergence(
        self,
        output_path: str = 'convergence.png',
        figsize: tuple = (14, 8),
        show: bool = False,
    ) -> str:
        """
        Vẽ đồ thị hội tụ:
        - Subplot trên: Chi phí hiện tại và Best cost theo iteration
        - Subplot dưới: Hệ số phạt λ theo iteration
        
        Returns:
            Đường dẫn file đã lưu
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib chưa được cài. Chạy: pip install matplotlib")
            return ''

        if not self.solver or not self.solver.convergence_history:
            print("Không có convergence history để plot")
            return ''

        history = self.solver.convergence_history
        iterations = [h['iteration'] for h in history]
        current_costs = [h['current_cost'] for h in history]
        best_costs = [h['best_cost'] for h in history]

        # Lấy lambda history
        lambda_keys = [k for k in history[0].keys() if k.startswith('lambda_')]

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        fig.suptitle(
            f'UTS Convergence – {self.problem.name}\n'
            f'Best: {min(h["best_cost"] for h in history):.2f}',
            fontsize=14, fontweight='bold'
        )

        # Subplot 1: Chi phí hội tụ
        ax1 = axes[0]
        ax1.plot(iterations, current_costs, alpha=0.5, linewidth=0.8,
                 color='steelblue', label='Current cost')
        ax1.plot(iterations, best_costs, linewidth=2,
                 color='darkred', label='Best cost')
        ax1.set_ylabel('Chi phí (khoảng cách)', fontsize=11)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Hội tụ chi phí', fontsize=11)

        # Đánh dấu điểm cải thiện
        improvements = [(h['iteration'], h['best_cost'])
                        for i, h in enumerate(history)
                        if i == 0 or h['best_cost'] < history[i-1]['best_cost']]
        if improvements:
            imp_iter, imp_cost = zip(*improvements)
            ax1.scatter(imp_iter, imp_cost, color='gold', s=30,
                       zorder=5, label='Cải thiện', marker='*')
            ax1.legend(loc='upper right')

        # Subplot 2: Lambda evolution
        ax2 = axes[1]
        for lk in lambda_keys:
            lambda_vals = [h.get(lk, 0) for h in history]
            constraint_name = lk.replace('lambda_', '')
            ax2.plot(iterations, lambda_vals, linewidth=1.5, label=f'λ_{constraint_name}')
        ax2.set_xlabel('Iteration', fontsize=11)
        ax2.set_ylabel('Hệ số phạt λ', fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_title('Strategic Oscillation – Biến thiên hệ số phạt', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        print(f"Đồ thị hội tụ đã lưu: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 2. ROUTE MAP (Matplotlib)
    # ------------------------------------------------------------------

    def plot_routes_matplotlib(
        self,
        output_path: str = 'routes.png',
        figsize: tuple = (12, 10),
        show: bool = False,
    ) -> str:
        """
        Vẽ bản đồ lộ trình bằng Matplotlib.
        
        Returns:
            Đường dẫn file đã lưu
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyArrowPatch
        except ImportError:
            print("matplotlib chưa được cài")
            return ''

        fig, ax = plt.subplots(figsize=figsize)
        nodes = self.problem.nodes

        # Vẽ depot
        depot = self.problem.depot
        ax.scatter(depot.x, depot.y, s=300, c='black', marker='s',
                   zorder=10, label='Depot')
        ax.annotate('Depot', (depot.x, depot.y),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)

        # Vẽ các routes
        active_routes = [r for r in self.solution.routes if not r.is_empty]
        for idx, route in enumerate(active_routes):
            color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]
            route_nodes = route.nodes

            # Vẽ customers
            for nid in route.customers:
                node = nodes[nid]
                ax.scatter(node.x, node.y, s=80, c=color, alpha=0.9, zorder=5)
                ax.annotate(str(nid), (node.x, node.y),
                            textcoords="offset points", xytext=(4, 4), fontsize=7)

            # Vẽ đường đi
            xs = [nodes[nid].x for nid in route_nodes]
            ys = [nodes[nid].y for nid in route_nodes]
            ax.plot(xs, ys, '-', color=color, alpha=0.7, linewidth=1.5,
                   label=f'Xe {route.vehicle.id} ({route.num_customers} điểm)')

            # Mũi tên chỉ hướng
            for i in range(len(route_nodes) - 1):
                src = nodes[route_nodes[i]]
                dst = nodes[route_nodes[i + 1]]
                mid_x = (src.x + dst.x) / 2
                mid_y = (src.y + dst.y) / 2
                dx = (dst.x - src.x) * 0.1
                dy = (dst.y - src.y) * 0.1
                ax.annotate('', xy=(mid_x + dx, mid_y + dy),
                            xytext=(mid_x - dx, mid_y - dy),
                            arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

        # Thống kê
        total_dist = self.solution.total_distance()
        title = (f'Lộ trình tối ưu – {self.problem.name}\n'
                 f'Tổng khoảng cách: {total_dist:.2f} | '
                 f'{len(active_routes)} xe | '
                 f'{len(self.problem.customers)} điểm giao')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        print(f"Bản đồ lộ trình đã lưu: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 3. FOLIUM INTERACTIVE MAP (dữ liệu thực tế)
    # ------------------------------------------------------------------

    def plot_routes_folium(
        self,
        output_path: str = 'routes_map.html',
        center_lat: float = 10.7769,   # Mặc định: TP.HCM
        center_lon: float = 106.7009,
    ) -> str:
        """
        Tạo bản đồ tương tác Folium cho dữ liệu thực tế.
        
        Sử dụng khi nodes có tọa độ lat/lon thực tế.
        
        Returns:
            Đường dẫn file HTML
        """
        try:
            import folium
        except ImportError:
            print("folium chưa được cài. Chạy: pip install folium")
            return ''

        m = folium.Map(location=[center_lat, center_lon], zoom_start=13)

        nodes = self.problem.nodes
        active_routes = [r for r in self.solution.routes if not r.is_empty]

        # Depot marker
        depot = self.problem.depot
        folium.Marker(
            location=[depot.y, depot.x],  # lat, lon
            popup=f'<b>Depot</b>',
            icon=folium.Icon(color='black', icon='home', prefix='fa'),
        ).add_to(m)

        # Routes
        for idx, route in enumerate(active_routes):
            color = ROUTE_COLORS[idx % len(ROUTE_COLORS)]

            # Customer markers
            for nid in route.customers:
                node = nodes[nid]
                folium.CircleMarker(
                    location=[node.y, node.x],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    popup=(f'<b>Node {nid}</b><br>'
                           f'Demand: {node.demand}<br>'
                           f'TW: [{node.ready_time:.0f}, {node.due_time:.0f}]'),
                ).add_to(m)

            # Route polyline
            coords = [[nodes[nid].y, nodes[nid].x] for nid in route.nodes]
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=3,
                opacity=0.8,
                tooltip=f'Xe {route.vehicle.id}: {route.num_customers} điểm',
            ).add_to(m)

        # Title
        title_html = f'''
        <div style="position: fixed; top: 10px; left: 50px; z-index: 1000;
             background: white; padding: 10px; border-radius: 5px; border: 2px solid grey;">
            <h4>{self.problem.name}</h4>
            <p>Tổng khoảng cách: {self.solution.total_distance():.2f}</p>
            <p>{len(active_routes)} xe | {len(self.problem.customers)} điểm giao</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        m.save(output_path)
        print(f"Bản đồ Folium đã lưu: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # 4. SOLUTION SUMMARY TABLE
    # ------------------------------------------------------------------

    def print_solution_table(self):
        """In bảng tóm tắt solution ra console."""
        print(f"\n{'='*65}")
        print(f"KẾT QUẢ GIẢI PHÁP – {self.problem.name}")
        print(f"{'='*65}")
        print(f"Tổng khoảng cách: {self.solution.total_distance():.2f}")
        print(f"{'─'*65}")
        print(f"{'Xe':>4} {'Nodes':>50} {'Load':>5} {'Dist':>7}")
        print(f"{'─'*65}")

        for route in self.solution.routes:
            if route.is_empty:
                continue
            load = route.total_load(self.problem)
            dist = route.total_distance(self.problem)
            nodes_str = ' → '.join(map(str, route.nodes))
            if len(nodes_str) > 48:
                nodes_str = nodes_str[:45] + '...'
            print(f"{route.vehicle.id:>4} {nodes_str:<50} {load:>5.0f} {dist:>7.2f}")

        print(f"{'─'*65}")
        print(f"Total: {self.solution.num_vehicles_used()} xe sử dụng\n")