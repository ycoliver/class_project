import coptpy as cp
from coptpy import COPT

# ==========================================
# 创建模型
# ==========================================
env = cp.Envr()
model = env.createModel("LGUAir_TwoStage_Dual")

# ==========================================
# 参数定义
# ==========================================
num_classes = 3      # 舱位等级数
num_scenarios = 3    # 场景数

# 空间占用系数
space_usage = {
    1: 2.0,   # 头等舱
    2: 1.5,   # 商务舱
    3: 1.0    # 经济舱
}

# 利润系数
profit = {
    1: 3.0,   # 头等舱
    2: 2.0,   # 商务舱
    3: 1.0    # 经济舱
}

# 总空间容量
total_capacity = 200

# 需求上限 demand[i, j]
demand = {
    (1, 1): 20,   (1, 2): 10,   (1, 3): 5,    # 头等舱
    (2, 1): 50,   (2, 2): 25,   (2, 3): 10,   # 商务舱
    (3, 1): 200,  (3, 2): 175,  (3, 3): 150   # 经济舱
}

# ==========================================
# 对偶变量定义
# ==========================================
# π: 对应空间约束的对偶变量
pi = model.addVar(
    lb=0.0,
    vtype=COPT.CONTINUOUS,
    name="pi"
)

# μ[i, j]: 对应座位限制约束 y_ij ≤ x_i 的对偶变量
mu = {}
for i in range(1, num_classes + 1):
    for j in range(1, num_scenarios + 1):
        mu[i, j] = model.addVar(
            lb=0.0,
            vtype=COPT.CONTINUOUS,
            name=f"mu_{i}_{j}"
        )

# λ[i, j]: 对应需求限制约束 y_ij ≤ d_ij 的对偶变量
lmbda = {}
for i in range(1, num_classes + 1):
    for j in range(1, num_scenarios + 1):
        lmbda[i, j] = model.addVar(
            lb=0.0,
            vtype=COPT.CONTINUOUS,
            name=f"lambda_{i}_{j}"
        )

# ==========================================
# 对偶目标函数（最小化）
# ==========================================
obj_expr = total_capacity * pi + cp.quicksum(
    demand[i, j] * lmbda[i, j]
    for i in range(1, num_classes + 1)
    for j in range(1, num_scenarios + 1)
)
model.setObjective(obj_expr, COPT.MINIMIZE)

# ==========================================
# 对偶约束
# ==========================================
# 1. 对应原问题变量 x_i 的对偶约束
#    原约束中 x_i 出现在：空间约束（系数 a_i）和座位限制（系数 -1 对每个 j）
#    对偶约束: a_i * π ≥ Σⱼ μ_ij
for i in range(1, num_classes + 1):
    model.addConstr(
        space_usage[i] * pi >= cp.quicksum(mu[i, j] for j in range(1, num_scenarios + 1)),
        name=f"dual_constr_x_{i}"
    )

# 2. 对应原问题变量 y_ij 的对偶约束
#    原约束中 y_ij 出现在：座位限制（系数 1）和需求限制（系数 1）
#    目标函数中系数为 r_i
#    对偶约束: μ_ij + λ_ij ≥ r_i
for i in range(1, num_classes + 1):
    for j in range(1, num_scenarios + 1):
        model.addConstr(
            mu[i, j] + lmbda[i, j] >= profit[i],
            name=f"dual_constr_y_{i}_{j}"
        )

# ==========================================
# 求解
# ==========================================
model.solve()

# ==========================================
# 输出结果
# ==========================================
print("=" * 70)
print("LGUAir两阶段问题 - 对偶问题（Dual of LP Relaxation）")
print("=" * 70)
print("\n【理论说明】")
print("• 原问题为整数规划（ILP）")
print("• 此对偶问题是原ILP的LP松弛的对偶")
print("• 对偶最优值是原ILP最优值的上界")
print("=" * 70)

if model.status == COPT.OPTIMAL:
    print(f"\n✓ 模型状态: 最优解")
    print(f"\n对偶最优值（上界）: W* = {model.objval:.4f}")
    
    # 空间约束的影子价格
    print("\n" + "=" * 70)
    print(f"空间约束的影子价格（Shadow Price）")
    print("=" * 70)
    print(f"π (空间容量的边际价值) = {pi.x:.4f}")
    print(f"\n解释: 每增加1单位空间容量，目标函数约增加 {pi.x:.4f}")
    
    # 座位限制约束的对偶变量
    print("\n" + "=" * 70)
    print("座位限制约束的对偶变量 μ_ij (y_ij ≤ x_i)")
    print("=" * 70)
    class_names = {1: "头等舱", 2: "商务舱", 3: "经济舱"}
    scenario_names = {1: "工作日早晚", 2: "周末", 3: "工作日午间"}
    
    for i in range(1, num_classes + 1):
        print(f"\n{class_names[i]}:")
        print(f"  {'场景':<15} {'μ_ij':<12} {'状态'}")
        print(f"  {'-'*40}")
        for j in range(1, num_scenarios + 1):
            status = "绑定" if mu[i, j].x > 1e-6 else "松弛"
            print(f"  {scenario_names[j]:<15} {mu[i, j].x:>10.4f}  {status}")
    
    # 需求限制约束的对偶变量
    print("\n" + "=" * 70)
    print("需求限制约束的对偶变量 λ_ij (y_ij ≤ d_ij)")
    print("=" * 70)
    
    for i in range(1, num_classes + 1):
        print(f"\n{class_names[i]}:")
        print(f"  {'场景':<15} {'λ_ij':<12} {'状态'}")
        print(f"  {'-'*40}")
        for j in range(1, num_scenarios + 1):
            status = "绑定" if lmbda[i, j].x > 1e-6 else "松弛"
            print(f"  {scenario_names[j]:<15} {lmbda[i, j].x:>10.4f}  {status}")
    
    # 对偶约束验证
    print("\n" + "=" * 70)
    print("对偶约束验证")
    print("=" * 70)
    
    # 验证约束1: a_i * π ≥ Σⱼ μ_ij
    print("\n约束1: 对应原问题变量 x_i")
    print(f"{'舱位':<12} {'a_i * π':<15} {'≥':<5} {'Σⱼ μ_ij':<15} {'状态'}")
    print("-" * 60)
    for i in range(1, num_classes + 1):
        lhs = space_usage[i] * pi.x
        rhs = sum(mu[i, j].x for j in range(1, num_scenarios + 1))
        status = "✓" if lhs >= rhs - 1e-6 else "✗"
        print(f"{class_names[i]:<10} {lhs:>12.4f} {'≥':^5} {rhs:>12.4f}   {status}")
    
    # 验证约束2: μ_ij + λ_ij ≥ r_i
    print("\n约束2: 对应原问题变量 y_ij")
    print(f"{'舱位-场景':<20} {'μ_ij + λ_ij':<15} {'≥':<5} {'r_i':<10} {'状态'}")
    print("-" * 60)
    for i in range(1, num_classes + 1):
        for j in range(1, num_scenarios + 1):
            lhs = mu[i, j].x + lmbda[i, j].x
            rhs = profit[i]
            status = "✓" if lhs >= rhs - 1e-6 else "✗"
            label = f"{class_names[i]}-场景{j}"
            print(f"{label:<18} {lhs:>12.4f} {'≥':^5} {rhs:>8.1f}     {status}")
    
    # 经济解释
    print("\n" + "=" * 70)
    print("经济解释")
    print("=" * 70)
    print(f"\n1. 影子价格 π = {pi.x:.4f}")
    print("   • 表示空间容量的边际价值")
    print("   • 每增加1个经济舱座位单位的容量，利润约增加π")
    
    print("\n2. 对偶变量 μ_ij:")
    print("   • 表示座位分配约束的机会成本")
    print("   • 若 μ_ij > 0: 该约束绑定，增加座位分配会提高利润")
    print("   • 若 μ_ij = 0: 该约束松弛，座位分配有余量")
    
    print("\n3. 对偶变量 λ_ij:")
    print("   • 表示需求限制的机会成本")
    print("   • 若 λ_ij > 0: 需求约束绑定，提高需求会增加利润")
    print("   • 若 λ_ij = 0: 需求有富余，提高需求无法增加利润")

else:
    print(f"\n✗ 模型状态: {model.status}")
    print("未找到最优解")

print("\n" + "=" * 70)