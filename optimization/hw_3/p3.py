import coptpy as cp
from coptpy import COPT

# ==========================================
# 创建模型
# ==========================================
env = cp.Envr()
model = env.createModel("LGUAir_TwoStage_Primal")

# ==========================================
# 参数定义
# ==========================================
num_classes = 3      # 舱位等级数
num_scenarios = 3    # 场景数

# 空间占用系数 (经济舱座位单位)
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

# 需求上限 demand[i, j]: 舱位i在场景j的需求
demand = {
    (1, 1): 20,   (1, 2): 10,   (1, 3): 5,    # 头等舱在各场景
    (2, 1): 50,   (2, 2): 25,   (2, 3): 10,   # 商务舱在各场景
    (3, 1): 200,  (3, 2): 175,  (3, 3): 150   # 经济舱在各场景
}

# ==========================================
# 决策变量
# ==========================================
# x[i]: 分配给舱位等级 i 的座位数（第一阶段决策）
x = {}
for i in range(1, num_classes + 1):
    x[i] = model.addVar(
        lb=0.0,
        vtype=COPT.INTEGER,
        name=f"x_{i}"
    )

# y[i, j]: 场景 j 中售出的舱位 i 的票数（第二阶段决策）
y = {}
for i in range(1, num_classes + 1):
    for j in range(1, num_scenarios + 1):
        y[i, j] = model.addVar(
            lb=0.0,
            vtype=COPT.INTEGER,
            name=f"y_{i}_{j}"
        )

# ==========================================
# 目标函数（最大化总利润）
# ==========================================
obj_expr = cp.quicksum(
    profit[i] * y[i, j]
    for i in range(1, num_classes + 1)
    for j in range(1, num_scenarios + 1)
)
model.setObjective(obj_expr, COPT.MAXIMIZE)

# ==========================================
# 约束条件
# ==========================================
# 1. 空间分配约束：总的座位空间不超过容量
space_constraint = cp.quicksum(
    space_usage[i] * x[i] for i in range(1, num_classes + 1)
)
model.addConstr(
    space_constraint <= total_capacity,
    name="total_space_constraint"
)

# 2. 座位限制约束：每个场景中的售票数不能超过分配的座位数
for i in range(1, num_classes + 1):
    for j in range(1, num_scenarios + 1):
        model.addConstr(
            y[i, j] <= x[i],
            name=f"seat_limit_{i}_{j}"
        )

# 3. 需求限制约束：售票数不能超过需求
for i in range(1, num_classes + 1):
    for j in range(1, num_scenarios + 1):
        model.addConstr(
            y[i, j] <= demand[i, j],
            name=f"demand_limit_{i}_{j}"
        )

# ==========================================
# 求解
# ==========================================
model.solve()

# ==========================================
# 输出结果
# ==========================================
print("=" * 70)
print("LGUAir两阶段座位分配问题 - 原问题（Primal）")
print("=" * 70)
print(f"总容量: {total_capacity} 经济舱座位单位")
print("=" * 70)

if model.status == COPT.OPTIMAL:
    print(f"\n✓ 模型状态: 最优解")
    print(f"\n最优目标值（总利润）: {model.objval:.2f}")
    
    # 输出第一阶段决策：座位分配
    print("\n" + "=" * 70)
    print("第一阶段决策：座位分配 (x_i)")
    print("=" * 70)
    class_names = {1: "头等舱", 2: "商务舱", 3: "经济舱"}
    
    total_space_used = 0
    print(f"{'舱位':<12} {'分配座位':<12} {'空间占用':<15} {'空间单位'}")
    print("-" * 70)
    for i in range(1, num_classes + 1):
        space = space_usage[i] * x[i].x
        total_space_used += space
        print(f"{class_names[i]:<10} {x[i].x:>10.0f} {space:>12.1f} {space_usage[i]:>12.1f}")
    
    print("-" * 70)
    print(f"{'总计':<10} {'':<10} {total_space_used:>12.1f} / {total_capacity}")
    print(f"空间利用率: {total_space_used/total_capacity*100:.2f}%")
    
    # 输出第二阶段决策：各场景售票情况
    print("\n" + "=" * 70)
    print("第二阶段决策：各场景售票数量 (y_ij)")
    print("=" * 70)
    scenario_names = {1: "工作日早晚", 2: "周末", 3: "工作日午间"}
    
    for j in range(1, num_scenarios + 1):
        print(f"\n场景 {j}: {scenario_names[j]}")
        print(f"  {'舱位':<12} {'售票数':<10} {'座位数':<10} {'需求':<10} {'利润贡献'}")
        print(f"  {'-'*60}")
        
        scenario_profit = 0
        for i in range(1, num_classes + 1):
            ticket_profit = profit[i] * y[i, j].x
            scenario_profit += ticket_profit
            print(f"  {class_names[i]:<10} {y[i, j].x:>8.0f} "
                  f"{x[i].x:>8.0f} {demand[i, j]:>8.0f} {ticket_profit:>10.2f}")
        
        print(f"  {'-'*60}")
        print(f"  场景利润: {scenario_profit:.2f}")
    
    # 约束满足情况
    print("\n" + "=" * 70)
    print("约束验证")
    print("=" * 70)
    
    # 空间约束
    space_used = sum(space_usage[i] * x[i].x for i in range(1, num_classes + 1))
    space_slack = total_capacity - space_used
    print(f"空间约束: {space_used:.1f} ≤ {total_capacity} (松弛量: {space_slack:.1f}) ✓")
    
    # 座位限制约束
    print("\n座位限制约束 (y_ij ≤ x_i):")
    for i in range(1, num_classes + 1):
        print(f"  {class_names[i]}:")
        for j in range(1, num_scenarios + 1):
            status = "✓" if y[i, j].x <= x[i].x + 1e-6 else "✗"
            print(f"    场景{j}: {y[i, j].x:.0f} ≤ {x[i].x:.0f} {status}")
    
    # 需求限制约束
    print("\n需求限制约束 (y_ij ≤ d_ij):")
    for i in range(1, num_classes + 1):
        print(f"  {class_names[i]}:")
        for j in range(1, num_scenarios + 1):
            status = "✓" if y[i, j].x <= demand[i, j] + 1e-6 else "✗"
            slack = demand[i, j] - y[i, j].x
            print(f"    场景{j}: {y[i, j].x:.0f} ≤ {demand[i, j]:.0f} "
                  f"(松弛: {slack:.0f}) {status}")

else:
    print(f"\n✗ 模型状态: {model.status}")
    print("未找到最优解")

print("\n" + "=" * 70)