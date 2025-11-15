import coptpy as cp
from coptpy import COPT

# 距离矩阵
d = {
    (1,2): 13, (1,3): 8, (1,4): 15, (1,5): 7,
    (2,1): 13, (2,3): 5, (2,4): 7, (2,5): 14,
    (3,1): 8, (3,2): 5, (3,4): 15, (3,5): 17,
    (4,1): 15, (4,2): 7, (4,3): 15, (4,5): 8,
    (5,1): 7, (5,2): 14, (5,3): 17, (5,4): 8
}

N = [1,2,3,4,5]

env = cp.Envr()
model = env.createModel("PCB_Drilling_TSP")

# 决策变量 x_ij
x = {}
for i in N:
    for j in N:
        if i != j:
            x[i,j] = model.addVar(vtype=COPT.BINARY, name=f"x_{i}_{j}")

# MTZ 辅助变量 u_i
u = {}
for i in N:
    if i != 1:
        u[i] = model.addVar(lb=1, ub=len(N)-1, vtype=COPT.CONTINUOUS, name=f"u_{i}")

# 目标函数
obj = cp.quicksum(d[i,j]*x[i,j] for i,j in x)
model.setObjective(obj, COPT.MINIMIZE)

# 每个孔正好离开一次
for i in N:
    model.addConstr(cp.quicksum(x[i,j] for j in N if j != i) == 1, name=f"leave_{i}")

# 每个孔正好进入一次
for j in N:
    model.addConstr(cp.quicksum(x[i,j] for i in N if i != j) == 1, name=f"enter_{j}")

# MTZ 子循环消除约束
for i in N:
    for j in N:
        if i != j and i != 1 and j != 1:
            model.addConstr(u[i] - u[j] + (len(N)-1)*x[i,j] <= len(N)-2, name=f"mtz_{i}_{j}")

# 求解
model.solve()

# 输出结果
if model.status == COPT.OPTIMAL:
    print("最优孔钻顺序及路径:")
    path = []
    current = 1
    visited = set([1])
    while len(path) < len(N):
        for j in N:
            if j != current and x[current,j].x > 0.5:
                path.append((current,j))
                current = j
                visited.add(j)
                break
    print(path)
    print(f"\n最短总距离: {model.objval} mm")
else:
    print("未找到最优解")
