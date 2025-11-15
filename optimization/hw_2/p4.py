import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import FancyArrowPatch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(1, 1, figsize=(14, 12))

# 定义坐标范围
x1 = np.linspace(0, 30, 500)

# 约束条件
# 1. x1 + 3x2 >= 15  =>  x2 >= (15 - x1) / 3
x2_c1 = (15 - x1) / 3

# 2. 2x1 + x2 >= 10  =>  x2 >= 10 - 2x1
x2_c2 = 10 - 2*x1

# 3. x1 + 2x2 <= 40  =>  x2 <= (40 - x1) / 2
x2_c3 = (40 - x1) / 2

# 4. 3x1 + x2 <= 60  =>  x2 <= 60 - 3x1
x2_c4 = 60 - 3*x1

# 绘制约束条件边界
ax.plot(x1, x2_c1, 'r-', linewidth=2.5, label=r'$x_1 + 3x_2 = 15$')
ax.plot(x1, x2_c2, 'b-', linewidth=2.5, label=r'$2x_1 + x_2 = 10$')
ax.plot(x1, x2_c3, 'g-', linewidth=2.5, label=r'$x_1 + 2x_2 = 40$')
ax.plot(x1, x2_c4, 'm-', linewidth=2.5, label=r'$3x_1 + x_2 = 60$')

# 绘制坐标轴约束
ax.axhline(y=0, color='black', linewidth=2, label=r'$x_2 = 0$')
ax.axvline(x=0, color='black', linewidth=2, label=r'$x_1 = 0$')

# 计算可行域顶点
# 需要找到所有约束的交点，然后筛选出可行的
vertices = []

# 约束交点
# C1 ∩ C2: x1 + 3x2 = 15, 2x1 + x2 = 10
# 解得: x1 = 3, x2 = 4
vertices.append((3, 4))

# C1 ∩ x2=0: x1 + 3(0) = 15 => x1 = 15
vertices.append((15, 0))

# C2 ∩ x2=0: 2x1 + 0 = 10 => x1 = 5
vertices.append((5, 0))

# C3 ∩ C4: x1 + 2x2 = 40, 3x1 + x2 = 60
# 解得: x1 = 16, x2 = 12
vertices.append((16, 12))

# C3 ∩ x2=0: x1 + 2(0) = 40 => x1 = 40 (需要检查是否满足其他约束)
# 检查: 3*40 = 120 > 60, 不满足C4
# 需要找 C4 ∩ x2=0: 3x1 + 0 = 60 => x1 = 20
vertices.append((20, 0))

# C1 ∩ C3: x1 + 3x2 = 15, x1 + 2x2 = 40
# 解得: x2 = 25, x1 = -60 (不可行)

# C2 ∩ C4: 2x1 + x2 = 10, 3x1 + x2 = 60
# 解得: x1 = 50, x2 = -90 (不可行)

# C1 ∩ C4: x1 + 3x2 = 15, 3x1 + x2 = 60
# 解得: x1 = 21.75, x2 = -2.25 (不可行)

# C2 ∩ C3: 2x1 + x2 = 10, x1 + 2x2 = 40
# 解得: x1 = -20/3 (不可行)

# 筛选可行顶点
feasible_vertices = []
for v in vertices:
    x1_v, x2_v = v
    # 检查所有约束
    if (x1_v >= 0 and x2_v >= 0 and
        x1_v + 3*x2_v >= 15 - 1e-6 and
        2*x1_v + x2_v >= 10 - 1e-6 and
        x1_v + 2*x2_v <= 40 + 1e-6 and
        3*x1_v + x2_v <= 60 + 1e-6):
        feasible_vertices.append(v)

# 按逆时针排序
def angle_from_center(point, center):
    return np.arctan2(point[1] - center[1], point[0] - center[0])

if feasible_vertices:
    center = np.mean(feasible_vertices, axis=0)
    feasible_vertices.sort(key=lambda p: angle_from_center(p, center))

# 绘制可行域
if len(feasible_vertices) > 2:
    polygon = Polygon(feasible_vertices, alpha=0.25, facecolor='lightblue', 
                      edgecolor='navy', linewidth=3, label='fesible region')
    ax.add_patch(polygon)

# 标记可行域顶点
for i, (x1_v, x2_v) in enumerate(feasible_vertices):
    obj_value = x1_v + x2_v
    ax.plot(x1_v, x2_v, 'ko', markersize=10)
    ax.annotate(f'({x1_v:.1f}, {x2_v:.1f})\nz={obj_value:.1f}', 
                xy=(x1_v, x2_v), xytext=(5, 5),
                textcoords='offset points', fontsize=9, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# 单纯形法迭代路径
iterations = [
    (3, 4, 0, "initial BFS", "B={x₁,x₂,x₅,x₆}"),
    (7, 0, 1, "iter 1", "B={x₁,x₄,x₅,x₆}"),
    (20, 0, 2, "iter 2", "B={x₁,x₄,x₅,x₃}"),
    (16, 12, 3, "iter 3 (optimal)", "B={x₁,x₄,x₂,x₃}")
]

# 绘制迭代点和路径
colors = ['red', 'orange', 'gold', 'green']
for i, (x1_v, x2_v, iteration, label, basis) in enumerate(iterations):
    obj_value = x1_v + x2_v
    
    # 绘制迭代点
    if i < len(iterations) - 1:
        ax.plot(x1_v, x2_v, 'o', color=colors[i], markersize=18, 
                markeredgecolor='black', markeredgewidth=2.5, zorder=5)
    else:
        # 最优解用星号
        ax.plot(x1_v, x2_v, '*', color=colors[i], markersize=30, 
                markeredgecolor='black', markeredgewidth=2.5, zorder=5)
    
    # 标注
    offset_x = 15 if x1_v < 15 else -80
    offset_y = 15 if i != 2 else -35
    ax.annotate(f'{label}\n({x1_v:.0f}, {x2_v:.0f})\n{basis}\nz = {obj_value:.0f}', 
                xy=(x1_v, x2_v), xytext=(offset_x, offset_y),
                textcoords='offset points', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], 
                         alpha=0.7, edgecolor='black', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3',
                               color='black', lw=2))

# 绘制迭代路径
for i in range(len(iterations) - 1):
    x1_from, x2_from = iterations[i][0], iterations[i][1]
    x1_to, x2_to = iterations[i+1][0], iterations[i+1][1]
    
    arrow = FancyArrowPatch((x1_from, x2_from), (x1_to, x2_to),
                           arrowstyle='->', mutation_scale=30, 
                           linewidth=3, color='darkblue', zorder=4,
                           linestyle='--')
    ax.add_patch(arrow)

# 绘制目标函数等高线
obj_values = [7, 14, 21, 28]
for obj_val in obj_values:
    x1_line = np.linspace(0, 30, 100)
    x2_line = obj_val - x1_line
    if obj_val == 28:
        ax.plot(x1_line, x2_line, 'g-', linewidth=3, alpha=0.8, 
                label=f'objection function: x₁+x₂={obj_val} (optimal)')
    else:
        ax.plot(x1_line, x2_line, 'gray', linewidth=1.5, alpha=0.5, linestyle=':')
        # 标注等高线
        ax.text(obj_val*0.5, obj_val*0.5 + 1, f'z={obj_val}', 
                fontsize=10, color='gray', style='italic')

# 设置图形属性
ax.set_xlim(-2, 30)
ax.set_ylim(-2, 22)
ax.set_xlabel(r'$x_1$', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$x_2$', fontsize=16, fontweight='bold')
ax.set_title('objection\nmax $z = x_1 + x_2$', 
             fontsize=18, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.set_aspect('equal', adjustable='box')

# 添加说明文本
info_text = """Iteration Process:
- Initial BFS: (3, 4), z=7
  Basic variables: {x₁, x₂, x₅, x₆}
  
- Iteration 1: x₄ enters, x₂ leaves
  New solution: (7, 0), z=7
  
- Iteration 2: x₃ enters, x₆ leaves
  New solution: (20, 0), z=20
  
- Iteration 3: x₂ enters, x₅ leaves
  Optimal solution: (16, 12), z*=28 ✓"""

ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
# plt.show()
plt.savefig('simplex_method_iterations.png', dpi=300)

# 输出详细信息
print("="*60)
print("单纯形法迭代过程详细信息")
print("="*60)
print("\n原问题:")
print("max z = x₁ + x₂")
print("s.t.")
print("  x₁ + 3x₂ ≥ 15")
print("  2x₁ + x₂ ≥ 10")
print("  x₁ + 2x₂ ≤ 40")
print("  3x₁ + x₂ ≤ 60")
print("  x₁, x₂ ≥ 0")

print("\n" + "="*60)
print("迭代详情:")
print("="*60)

for x1_v, x2_v, iteration, label, basis in iterations:
    obj_value = x1_v + x2_v
    print(f"\n{label}:")
    print(f"  基变量集合: {basis}")
    print(f"  当前解: (x₁, x₂) = ({x1_v:.0f}, {x2_v:.0f})")
    print(f"  目标函数值: z = {obj_value:.0f}")
    
    # 验证约束
    c1 = x1_v + 3*x2_v
    c2 = 2*x1_v + x2_v
    c3 = x1_v + 2*x2_v
    c4 = 3*x1_v + x2_v
    print(f"  约束验证:")
    print(f"    x₁ + 3x₂ = {c1:.1f} {'≥' if c1 >= 15 else '<'} 15 {'✓' if c1 >= 15-1e-6 else '✗'}")
    print(f"    2x₁ + x₂ = {c2:.1f} {'≥' if c2 >= 10 else '<'} 10 {'✓' if c2 >= 10-1e-6 else '✗'}")
    print(f"    x₁ + 2x₂ = {c3:.1f} {'≤' if c3 <= 40 else '>'} 40 {'✓' if c3 <= 40+1e-6 else '✗'}")
    print(f"    3x₁ + x₂ = {c4:.1f} {'≤' if c4 <= 60 else '>'} 60 {'✓' if c4 <= 60+1e-6 else '✗'}")

print("\n" + "="*60)
print(f"最优解: (x₁*, x₂*) = (16, 12)")
print(f"最优目标值: z* = 28")
print("="*60)