import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.optimize import linprog

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

# 定义x1的范围
x1 = np.linspace(-1, 10, 400)

# 绘制约束条件直线
# 约束1: -x1 + x2 <= 2.5  =>  x2 <= x1 + 2.5
x2_constraint1 = x1 + 2.5
ax.plot(x1, x2_constraint1, 'r-', linewidth=2, label=r'$-x_1 + x_2 = 2.5$')
ax.fill_between(x1, -1, x2_constraint1, alpha=0.1, color='red')

# 约束2: x1 + 2x2 <= 9  =>  x2 <= (9 - x1) / 2
x2_constraint2 = (9 - x1) / 2
ax.plot(x1, x2_constraint2, 'b-', linewidth=2, label=r'$x_1 + 2x_2 = 9$')
ax.fill_between(x1, -1, x2_constraint2, alpha=0.1, color='blue')

# 约束3: 0 <= x1 <= 4
ax.axvline(x=0, color='g', linewidth=2, label=r'$x_1 = 0$')
ax.axvline(x=4, color='purple', linewidth=2, label=r'$x_1 = 4$')
ax.axvspan(-1, 0, alpha=0.1, color='green')
ax.axvspan(4, 5, alpha=0.1, color='purple')

# 约束4: 0 <= x2 <= 3
ax.axhline(y=0, color='orange', linewidth=2, label=r'$x_2 = 0$')
ax.axhline(y=3, color='brown', linewidth=2, label=r'$x_2 = 3$')
ax.axhspan(-1, 0, alpha=0.1, color='orange')
ax.axhspan(3, 5, alpha=0.1, color='brown')

# 找到可行域的顶点
vertices = []

# 顶点1: x1=0, x2=0
vertices.append((0, 0))

# 顶点2: x1=4, x2=0
vertices.append((4, 0))

# 顶点3: x1=4, x2=2.5 (x1=4 与 x1+2x2=9 的交点)
x1_v = 4
x2_v = (9 - x1_v) / 2
if 0 <= x2_v <= 3:
    vertices.append((x1_v, x2_v))

# 顶点4: x1+2x2=9 与 x2=3 的交点
x2_v = 3
x1_v = 9 - 2 * x2_v
if 0 <= x1_v <= 4:
    vertices.append((x1_v, x2_v))

# 顶点5: -x1+x2=2.5 与 x2=3 的交点
x2_v = 3
x1_v = x2_v - 2.5
if 0 <= x1_v <= 4:
    vertices.append((x1_v, x2_v))

# 顶点6: x1=0 与 x2=3 的交点
vertices.append((0, 3))

# 顶点7: x1=0 与 -x1+x2=2.5 的交点
x1_v = 0
x2_v = x1_v + 2.5
if 0 <= x2_v <= 3:
    vertices.append((x1_v, x2_v))

# 去重并排序顶点（按逆时针）
vertices = list(set(vertices))

# 计算可行域顶点（需要满足所有约束）
feasible_vertices = []
for v in vertices:
    x1_v, x2_v = v
    # 检查所有约束
    if (0 <= x1_v <= 4 and 
        0 <= x2_v <= 3 and 
        -x1_v + x2_v <= 2.5 + 1e-6 and 
        x1_v + 2*x2_v <= 9 + 1e-6):
        feasible_vertices.append(v)

# 按逆时针排序
def angle_from_center(point, center):
    return np.arctan2(point[1] - center[1], point[0] - center[0])

if feasible_vertices:
    center = np.mean(feasible_vertices, axis=0)
    feasible_vertices.sort(key=lambda p: angle_from_center(p, center))

# 绘制可行域
if len(feasible_vertices) > 2:
    polygon = Polygon(feasible_vertices, alpha=0.3, facecolor='yellow', 
                      edgecolor='black', linewidth=3, label='fesible region')
    ax.add_patch(polygon)

# 标记顶点并计算目标函数值
print("可行域顶点及目标函数值 (x₃ = x₁ + x₂):")
print("-" * 50)
max_value = -np.inf
optimal_point = None

for i, (x1_v, x2_v) in enumerate(feasible_vertices):
    x3_v = x1_v + x2_v  # 目标函数值
    ax.plot(x1_v, x2_v, 'ro', markersize=12, markeredgecolor='black', markeredgewidth=2)
    ax.annotate(f'V{i+1}({x1_v:.2f}, {x2_v:.2f})\nx₃={x3_v:.2f}', 
                xy=(x1_v, x2_v), xytext=(10, 10),
                textcoords='offset points', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    print(f"顶点 {i+1}: x₁ = {x1_v:.2f}, x₂ = {x2_v:.2f}, x₃ = {x3_v:.2f}")
    
    if x3_v > max_value:
        max_value = x3_v
        optimal_point = (x1_v, x2_v, x3_v)

# 绘制目标函数的等高线（几条）
for c in np.linspace(0, max_value + 2, 8):
    # x1 + x2 = c  =>  x2 = c - x1
    x1_obj = np.linspace(-1, 10, 100)
    x2_obj = c - x1_obj
    if c == 0:
        ax.plot(x1_obj, x2_obj, 'k--', linewidth=1, alpha=0.3)
    else:
        ax.plot(x1_obj, x2_obj, 'k--', linewidth=1, alpha=0.5, label=f'$x_1+x_2={c:.1f}$' if c == max_value else '')

# 标记最优解
if optimal_point:
    ax.plot(optimal_point[0], optimal_point[1], 'g*', markersize=25, 
            markeredgecolor='black', markeredgewidth=2, label='optimal result', zorder=5)
    print("-" * 50)
    print(f"\noptimal result:")
    print(f"x₁* = {optimal_point[0]:.2f}")
    print(f"x₂* = {optimal_point[1]:.2f}")
    print(f"x₃* = {optimal_point[2]:.2f} (最大值)")

# 设置图形属性
ax.set_xlim(-0.5, 10)
ax.set_ylim(-0.5, 5)
ax.set_xlabel(r'$x_1$', fontsize=14)
ax.set_ylabel(r'$x_2$', fontsize=14)
ax.set_title('objection: max $x_3 = x_1 + x_2$', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', fontsize=10)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
# plt.show()
plt.savefig('linear_programming_solution.png', dpi=300)

# 验证结果
print("\n" + "="*50)
print("验证约束条件（最优解处）:")
print("="*50)
if optimal_point:
    x1_opt, x2_opt, x3_opt = optimal_point
    print(f"x₁ + x₂ - x₃ = {x1_opt + x2_opt - x3_opt:.6f} (应该 = 0) ✓")
    print(f"-x₁ + x₂ = {-x1_opt + x2_opt:.2f} (应该 ≤ 2.5) {'✓' if -x1_opt + x2_opt <= 2.5 + 1e-6 else '✗'}")
    print(f"x₁ + 2x₂ = {x1_opt + 2*x2_opt:.2f} (应该 ≤ 9) {'✓' if x1_opt + 2*x2_opt <= 9 + 1e-6 else '✗'}")
    print(f"0 ≤ x₁ = {x1_opt:.2f} ≤ 4 {'✓' if 0 <= x1_opt <= 4 else '✗'}")
    print(f"0 ≤ x₂ = {x2_opt:.2f} ≤ 3 {'✓' if 0 <= x2_opt <= 3 else '✗'}")