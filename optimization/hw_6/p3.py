import numpy as np

# Define problem
A = np.array([[2, 0], [0, 5]])
def f(x):
    return x.T @ A @ x
def grad_f(x):
    return 2 * A @ x

# Initial point
x0 = np.array([1.0, 1.0])

# 1. Constant Step Size (alpha = 0.1)
print("--- Constant Step Size (alpha=0.1) ---")
x = x0.copy()
for k in range(10):
    g = grad_f(x)
    x = x - 0.1 * g
    print(f"Iter {k+1}: x = {x}, f(x) = {f(x):.4f}")

# 2. Exact Line Search
print("\n--- Exact Line Search ---")
x = x0.copy()
for k in range(10):
    g = grad_f(x)
    # Formula for exact alpha derived manually:
    # alpha = (g.T @ A @ x) / (g.T @ A @ g) ?? 
    # Let's use the explicit formula derived in Part A specific to this problem structure
    # Numerator of derivative: 4*g[0]*x[0] + 10*g[1]*x[1] ... wait simpler to use standard formula
    # For Quadratic f(x) = x^T A x, optimal alpha = (g^T g) / (2 g^T A g)
    num = g.T @ g
    den = 2 * g.T @ A @ g
    alpha = num / den
    
    x = x - alpha * g
    print(f"Iter {k+1}: alpha={alpha:.4f}, x = {x}, f(x) = {f(x):.4f}")

# 3. Backtracking Line Search
print("\n--- Backtracking Line Search ---")
x = x0.copy()
gamma = 0.5
sigma = 0.2
for k in range(10):
    g = grad_f(x)
    alpha = 1.0
    f_curr = f(x)
    norm_g_sq = np.linalg.norm(g)**2
    
    # Backtracking loop
    while True:
        x_new = x - alpha * g
        if f(x_new) <= f_curr - gamma * alpha * norm_g_sq:
            break
        alpha *= sigma
        
    x = x_new
    print(f"Iter {k+1}: alpha={alpha:.4f}, x = {x}, f(x) = {f(x):.4f}")