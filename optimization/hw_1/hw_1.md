# Q1

## A1
### **a) Linear Programming Formulation**

Let $x_1, x_2$ represent the production quantities of products A and B, respectively.

#### Objective Function

$y = \min (-9x_1 - 8x_2 + 1.2x_1 + 0.9x_2) = \min (-7.8x_1 - 7.1x_2)$

#### Constraints
$$\begin{cases}
x_1/4 + x_2/3 \leq 90 \\
x_1/8 + x_2/3 \leq 80 \\
x_1, x_2 \geq 0
\end{cases}$$


### **b) Adding Conditions**

Condition (i) is easy to incorporate. Let $z$ represent the overtime hours.

#### Objective Function

$y = \min (-7.8x_1 - 7.1x_2 + 7z)$

#### Constraints
$$
x_1/4 + x_2/3 \leq 90 + z \\
x_1/8 + x_2/3 \leq 80 \\
x_1, x_2 \geq 0 \\
0 \leq z \leq 50
$$

For condition (ii), it cannot be incorporated directly into the original linear equations, but the problem can be split into two subproblems, and the smaller optimal value of the subproblems can be taken.

#### **Subproblem 1**

#### Objective Function
$y = \min (-7.8x_1 - 7.1x_2)$

#### Constraints
$$
x_1/4 + x_2/3 \leq 90 \\
x_1/8 + x_2/3 \leq 80 \\
1.2x_1 + 0.9x_2 \leq 300 \\
x_1, x_2 \geq 0
$$

#### **Subproblem 2**

#### Objective Function
$y = \min (-9x_1 + 1.2*0.9x_1 + 8x_2 - 0.9*0.9x_2) = \min(-7.92x_1 - 7.19x_2)$

#### Constraints
$$
x_1/4 + x_2/3 \leq 90 \\
x_1/8 + x_2/3 \leq 80 \\
1.2x_1 + 0.9x_2 > 300 \\
x_1, x_2 \geq 0 
$$



# Q2

## A2
Python implementation can be found in p2.py

### **a) Linear Programming Formulation**

#### Objective: Minimize the sum of absolute deviations

#### Decision Variables: Slope $m \geq 0$, Intercept $b \geq 0$

#### Auxiliary Variables: For each data point, introduce $u_k \geq 0$ and $v_k \geq 0$, representing the positive and negative deviations of the k-th point

#### Objective Function
$min \sum(u_k + v_k)$

#### Constraints:

$$
\begin{cases}
m \cdot power_k + b + u_k - v_k = y_k \\
u_k, v_k \geq 0
\end{cases}
$$

### **b) Optimal Parameters and Results**

Using the code in p2.py, the estimated parameters minimizing the absolute deviations are $m = 3.44$ and $b = 28.95$.

Predicted values: $[63.36, 56.48, 73.68, 80.56, 59.92]$



## Q3

### A3

#### Decision Variables

$x_{ijk}$ represents the sales quantity from plant $i$ to outlet $j$ in period $k$.

$z_{ik}$ represents the quantity used from the inventory of plant $i$ in period $k$ from the previous period.

#### Auxiliary Variables

$f_{ij}$ represents the shipping cost per unit from plant $i$ to outlet $j$.

$p_{ik}$ represents the production cost of plant $i$ in period $k$.

$s_{ijk}$ represents the profit from selling at outlet $j$ from plant $i$ in period $k$.

$cap_{ik}$ represents the maximum production capacity of plant $i$ in period $k$.

#### Objective Function
$\min \sum_{i}\sum_{j}\sum_{k} x_{ijk}(f_{ij} + p_{ik} - s_{ijk}) + z_{ik}(f_{ij} + p_{ik} - s_{ijk}) + z_{ik}$

#### Constraints

$$
\begin{cases}
\sum_{j} x_{ijk} + z_{ik} \leq cap_{ik} + z_{i,k-1} \\
z_{i0} = 0 \\
0 \leq z_{i1} \leq 50 \\
z_{i2} = 0 \\
x_{ijk} \geq 0
\end{cases}
$$


# Q4

## A4
Python implementation can be found in p4.py

### **a) Network Diagram Design**

#### Node Definitions:
- Source Node (0): Represents the starting point of production capacity.
- Production Nodes (1,2,3): Correspond to production in periods 1, 2, and 3.
- Inventory Nodes (4,5,6): Represent storage of products across periods.
- Sales Nodes (7-15): Three sales outlets (A, B, C) for each period.
- Sink Node (16): Represents the final destination of sales.

#### Edge Design (reflecting “soft” demands):
- Source → Production Nodes: capacity = production capacity, cost = 0.
- Production Nodes → Inventory Nodes: capacity = production capacity, cost = production cost.
- Inventory Nodes → Sales Nodes: capacity = maximum sales, cost = shipping cost.
- Sales Nodes → Sink Node: capacity = maximum sales, cost = negative selling price (representing revenue).
- Inventory Nodes → Next Period Inventory: capacity = 100, cost = storage cost.


**The Flow plot is shown in plot.png**

### **b) Minimum Cost Network Flow Model**

Let:
- $p_j$ be the production quantity in period $j$.
- $I_i$ be the inventory at the end of period $i$.
- $x_{ij}$ be the sales quantity from inventory to sales outlet $i$ in period $j$.
- $c_i^p$ be the unit production cost in period $i$.
- $c_i^I$ be the unit storage cost in period $i$.
- $c_{ij}^s$ be the shipping cost from inventory to sales outlet $i$ in period $j$.
- $c_{ij}^v$ be the selling price at sales outlet $i$ in period $j$.

#### **Objective Function**
$$
y = \min \sum_j (p_j \cdot c_j^p) + \sum_i (I_i \cdot c_i^I) + \sum_i \sum_j (x_{ij} \cdot c_{ij}^s) - \sum_i \sum_j (x_{ij} \cdot c_{ij}^v)
$$

#### **Constraints**
- Production Capacity: $p_i \leq cap_i$
- Inventory Capacity: $I_i \leq 100$
- Sales Limit: $x_{ij} \leq capx_{ij}$
- Inventory Balance: $I_{i-1} + p_i = \sum_i(x_{ij}) + I_i, \quad i = 1,2$
- Inventory Capacity: $I_i \leq 100$

#### Decision Variables
$$
p_i \geq 0, \quad I_i \geq 0, \quad x_{ij} \geq 0
$$

### **c) Results**

- Total units sold: 530 units
- Minimum cost: \$-1410 (negative value represents net profit)

#### Economic Interpretation:
By optimizing production and inventory strategies, the company sold 530 units over three periods, achieving a total profit of \$1410 (absolute value of minimum cost).


# Q5

## A5

### Decision Variables
$x_t$ represents the number of clerks starting a shift at hour $t$.

$y_{ti}$ represents the number of clerks starting a shift at hour $t$ who take lunch during hour $i$.

### Objective Function
$$
\min \sum c_t x_t
$$

### Constraints
$$
\begin{cases}
r_i \leq \sum_t (x_t - y_{ti}) & \forall i \\
y_{ti} = 0 & \text{if } (i - t > 0 \text{ and } 3 \le i - t \le 5) \text{ or } (i - t < 0 \text{ and } 3 \le i + 24 - t \le 5)
\end{cases}
$$
