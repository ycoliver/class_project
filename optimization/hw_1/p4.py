import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx



class MinCostMaxFlow:
    def __init__(self, n):
        self.n = n  # 节点数
        self.graph = defaultdict(list)
        self.cost = defaultdict(dict)
        self.capacity = defaultdict(dict)

    def add_edge(self, u, v, cap, c):
        self.graph[u].append(v)
        self.graph[v].append(u)  # 反向边
        self.capacity[u][v] = cap
        self.capacity[v][u] = 0  # 反向边容量为0
        self.cost[u][v] = c
        self.cost[v][u] = -c  # 反向边费用为负

    def min_cost_max_flow(self, source, sink):
        flow = 0
        cost = 0

        while True:
            dist = [float('inf')] * self.n
            dist[source] = 0
            parent = [-1] * self.n
            in_queue = [False] * self.n
            queue = [(0, source)]
            while queue:
                d, u = heapq.heappop(queue)
                if in_queue[u]:
                    continue
                in_queue[u] = True
                for v in self.graph[u]:
                    if self.capacity[u][v] > 0 and dist[v] > dist[u] + self.cost[u][v]:
                        dist[v] = dist[u] + self.cost[u][v]
                        parent[v] = u
                        heapq.heappush(queue, (dist[v], v))

            if dist[sink] == float('inf'):
                break  # 找不到增广路径

            # 查找增广流量的最小容量
            flow_amount = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                flow_amount = min(flow_amount, self.capacity[u][v])
                v = u

            # 调整流量
            v = sink
            while v != source:
                u = parent[v]
                self.capacity[u][v] -= flow_amount
                self.capacity[v][u] += flow_amount
                cost += flow_amount * self.cost[u][v]
                v = u

            flow += flow_amount

        return flow, cost
    

def draw_mcmf_graph(mcmf, node_labels=None, figsize=(16, 12), show_zero_capacity=False):
    """
    绘制最小费用最大流图的有向图
    
    参数:
    mcmf: MinCostMaxFlow对象
    node_labels: 节点标签字典，如 {0: 'Source', 1: 'P1', ...}
    figsize: 图形大小
    show_zero_capacity: 是否显示容量为0的边（反向边）
    """
    # 创建有向图
    G = nx.DiGraph()
    
    # 添加所有节点
    for i in range(mcmf.n):
        G.add_node(i)
    
    # 添加边（只添加有正容量的边，除非指定显示零容量边）
    edge_labels = {}
    for u in mcmf.graph:
        for v in mcmf.graph[u]:
            capacity = mcmf.capacity[u][v]
            cost = mcmf.cost[u][v]
            
            if capacity > 0 or show_zero_capacity:
                G.add_edge(u, v)
                # 边标签：容量/费用
                edge_labels[(u, v)] = f"{capacity}/{cost}"
    
    # 设置图形
    plt.figure(figsize=figsize)
    
    # 为不同类型的节点设置不同颜色
    node_colors = []
    for i in range(mcmf.n):
        if i == 0:  # 源点
            node_colors.append('lightgreen')
        elif i == mcmf.n - 1:  # 汇点
            node_colors.append('lightcoral')
        elif 1 <= i <= 3:  # 生产节点
            node_colors.append('lightblue')
        elif 4 <= i <= 6:  # 库存节点
            node_colors.append('lightgray')
        else:  # 存储节点
            node_colors.append('lightpink')
    
    # 创建布局
    # 手动设置位置以获得更好的可视化效果
    pos = {}
    
    # 源点
    pos[0] = (0, 2)
    
    # 生产节点 (P1, P2, P3)
    for i in range(1, 4):
        pos[i] = (1, i-1)
    
    # 库存节点 (I1, I2, I3)
    for i in range(4, 7):
        pos[i] = (2, i-4)
    
    # 存储节点 (S1-S9, 纵向排列)
    for i in range(7, 16):
        period = (i - 7) // 3  # 0, 1, 2 (第几个周期)
        storage = (i - 7) % 3   # 0, 1, 2 (周期内第几个存储)
        pos[i] = (3 + period * 1.2, storage - 1)  # X轴按周期分开，Y轴纵向排列存储节点
    
    # 汇点
    pos[16] = (5, 1)
    
    # 绘制图
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=800, alpha=0.9)
    
    # 绘制边
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=20, 
                          arrowstyle='->', alpha=0.6)
    
    # 添加节点标签
    if node_labels:
        nx.draw_networkx_labels(G, pos, node_labels, font_size=10)
    else:
        nx.draw_networkx_labels(G, pos, font_size=10)
    
    # 添加边标签
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
    
    plt.title("network diagram", fontsize=14)
    plt.axis('off')
    
    # 添加图例
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                   markersize=10, label='Source'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=10, label='manufacturingplant'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                   markersize=10, label='store'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightpink', 
                   markersize=10, label='sales outlets'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=10, label='terminal')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig('plot.png', dpi=300, bbox_inches='tight')


mcmf = MinCostMaxFlow(17)



cap_p = [180,200,150]
cost_p = [8,10,11]
cap_i = [100,100,100]
cost_i = [1,1,1]
cap_s = [[50,100,75],[75,100,75],[20,80,50]]
cost_s = [[4,6,8],[4,6,8],[4,6,8]]
cap_t = [[50,100,75],[75,100,75],[20,80,50]]
cost_t = [[-15,-20,-13],[-19,-16,-21],[-15,-18,-18]]

o_index = 0
p_index = 1
i_index = 4
s_index = 7
t_index = 16
# 起始节点初始化
mcmf.add_edge(0,1,180,0)
mcmf.add_edge(0,2,200,0)
mcmf.add_edge(0,3,150,0)

for i in range(3):
    mcmf.add_edge(p_index+i,i_index+i,cap_p[i],cost_p[i])
    for j in range(3):
        mcmf.add_edge(i+i_index,3*i+j+s_index,cap_s[i][j],cost_s[i][j])
        mcmf.add_edge(3*i+j+s_index,t_index,cap_t[i][j],cost_t[i][j])
    if i != 2:

        mcmf.add_edge(i+i_index,i+i_index+1,cap_i[i],cost_i[i])
# mcmf.add_edge(6,t_index,50,2**32-1) # 最后一个周期的库存到销售的成本无限大

# 创建节点标签
node_labels = {
    0: 'Source',
    1: 'P1', 2: 'P2', 3: 'P3',  # 生产节点
    4: 'I1', 5: 'I2', 6: 'I3',  # 库存节点
    7: 'S1-1', 8: 'S1-2', 9: 'S1-3',    # 第一期存储
    10: 'S2-1', 11: 'S2-2', 12: 'S2-3',  # 第二期存储
    13: 'S3-1', 14: 'S3-2', 15: 'S3-3',  # 第三期存储
    16: 'Sink'
}

# 绘制图形
draw_mcmf_graph(mcmf, node_labels=node_labels)

max_flow, min_cost = mcmf.min_cost_max_flow(0, 16)
print(f"Maximize production: {max_flow}, Minimize Cost: {min_cost}")
