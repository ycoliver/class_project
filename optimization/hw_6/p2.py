import numpy as np

# Define the matrix A and objective function
A = np.array([[2, 0], [0, 5]])

def f(x):
    """Objective function: f(x) = x^T A x"""
    return x.T @ A @ x

def grad_f(x):
    """Gradient of f: ∇f(x) = 2Ax"""
    return 2 * A @ x

# Initial point
x0 = np.array([1.0, 1.0])

print("=" * 60)
print("Optimization Problem: min f(x) = x^T A x")
print(f"Matrix A:\n{A}")
print(f"Initial point x0 = {x0}")
print(f"f(x0) = {f(x0)}")
print(f"∇f(x0) = {grad_f(x0)}")
print("=" * 60)

# ============================================================
# Part (a): Manual Calculations for One Step
# ============================================================
print("\n" + "=" * 60)
print("Part (a): One Step of Gradient Descent (Manual Calculation)")
print("=" * 60)

# Strategy 1: Constant step size α = 0.1
print("\n--- Strategy 1: Constant Step Size α = 0.1 ---")
alpha_const = 0.1
x = x0.copy()
grad = grad_f(x)
x_new = x - alpha_const * grad
print(f"x0 = {x}")
print(f"∇f(x0) = {grad}")
print(f"x1 = x0 - 0.1 * ∇f(x0) = {x} - 0.1 * {grad} = {x_new}")
print(f"f(x1) = {f(x_new)}")

# Strategy 2: Exact line search
print("\n--- Strategy 2: Exact Line Search ---")
x = x0.copy()
grad = grad_f(x)
# For f(x) = x^T A x, the exact line search gives:
# α* = (∇f^T ∇f) / (2 * ∇f^T A ∇f)
alpha_exact = (grad.T @ grad) / (2 * grad.T @ A @ grad)
x_new_exact = x - alpha_exact * grad
print(f"x0 = {x}")
print(f"∇f(x0) = {grad}")
print(f"α* = (∇f^T ∇f) / (2 * ∇f^T A ∇f)")
print(f"   = ({grad.T @ grad}) / (2 * {grad.T @ A @ grad})")
print(f"   = {alpha_exact:.6f}")
print(f"x1 = x0 - α* * ∇f(x0) = {x_new_exact}")
print(f"f(x1) = {f(x_new_exact)}")

# Strategy 3: Backtracking line search with Armijo condition
print("\n--- Strategy 3: Backtracking Line Search (Armijo) ---")
print("Parameters: γ = 0.1, σ = 0.2")
gamma = 0.5
sigma = 0.2
x = x0.copy()
grad = grad_f(x)
alpha = 1.0  # Initial step size

print(f"x0 = {x}")
print(f"∇f(x0) = {grad}")
print(f"||∇f(x0)||^2 = {np.linalg.norm(grad)**2}")
print(f"\nArmijo condition: f(x - α∇f) ≤ f(x) - γ·α·||∇f||^2")
print(f"                  f(x - α∇f) ≤ {f(x)} - 0.1·α·{np.linalg.norm(grad)**2}")

iteration = 0
while True:
    x_trial = x - alpha * grad
    lhs = f(x_trial)
    rhs = f(x) - gamma * alpha * np.linalg.norm(grad)**2
    print(f"\nα = {alpha:.6f}:")
    print(f"  x_trial = {x_trial}")
    print(f"  f(x_trial) = {lhs:.6f}")
    print(f"  f(x) - γ·α·||∇f||^2 = {rhs:.6f}")
    
    if lhs <= rhs:
        print(f"  Armijo condition SATISFIED!")
        break
    else:
        print(f"  Armijo condition NOT satisfied ({lhs:.6f} > {rhs:.6f})")
        alpha = sigma * alpha
        print(f"  Update: α ← σ·α = {alpha:.6f}")
    iteration += 1
    if iteration > 20:
        break

x_new_bt = x - alpha * grad
print(f"\nFinal α = {alpha:.6f}")
print(f"x1 = {x_new_bt}")
print(f"f(x1) = {f(x_new_bt)}")

# ============================================================
# Part (b): 10 Iterations with Python
# ============================================================
print("\n" + "=" * 60)
print("Part (b): 10 Iterations of Gradient Descent")
print("=" * 60)

def gradient_descent_constant(x0, num_iters=10, alpha=0.1):
    """Gradient descent with constant step size"""
    x = x0.copy()
    history = [(0, x.copy(), f(x), alpha)]
    
    for k in range(num_iters):
        grad = grad_f(x)
        x = x - alpha * grad
        history.append((k+1, x.copy(), f(x), alpha))
    
    return x, history

def gradient_descent_exact(x0, num_iters=10):
    """Gradient descent with exact line search"""
    x = x0.copy()
    history = [(0, x.copy(), f(x), None)]
    
    for k in range(num_iters):
        grad = grad_f(x)
        if np.linalg.norm(grad) < 1e-12:
            break
        # Exact line search: α* = (∇f^T ∇f) / (2 * ∇f^T A ∇f)
        alpha = (grad.T @ grad) / (2 * grad.T @ A @ grad)
        x = x - alpha * grad
        history.append((k+1, x.copy(), f(x), alpha))
    
    return x, history

def gradient_descent_backtracking(x0, num_iters=10, gamma=0.5, sigma=0.2):
    """Gradient descent with backtracking line search (Armijo condition)"""
    x = x0.copy()
    history = [(0, x.copy(), f(x), None)]
    
    for k in range(num_iters):
        grad = grad_f(x)
        if np.linalg.norm(grad) < 1e-12:
            break
        
        # Backtracking line search
        alpha = 1.0
        while f(x - alpha * grad) > f(x) - gamma * alpha * np.linalg.norm(grad)**2:
            alpha = sigma * alpha
            if alpha < 1e-12:
                break
        
        x = x - alpha * grad
        history.append((k+1, x.copy(), f(x), alpha))
    
    return x, history

# Run all three methods
print("\n--- Method 1: Constant Step Size (α = 0.1) ---")
x_final_const, hist_const = gradient_descent_constant(x0, num_iters=10, alpha=0.1)
print(f"{'Iter':<6} {'x1':<12} {'x2':<12} {'f(x)':<15} {'α':<10}")
print("-" * 55)
for k, x, fx, alpha in hist_const:
    print(f"{k:<6} {x[0]:<12.6f} {x[1]:<12.6f} {fx:<15.6f} {alpha:<10.4f}")

print("\n--- Method 2: Exact Line Search ---")
x_final_exact, hist_exact = gradient_descent_exact(x0, num_iters=10)
print(f"{'Iter':<6} {'x1':<12} {'x2':<12} {'f(x)':<15} {'α':<10}")
print("-" * 55)
for k, x, fx, alpha in hist_exact:
    alpha_str = f"{alpha:.6f}" if alpha is not None else "N/A"
    print(f"{k:<6} {x[0]:<12.6f} {x[1]:<12.6f} {fx:<15.6f} {alpha_str:<10}")

print("\n--- Method 3: Backtracking Line Search (γ=0.5, σ=0.2) ---")
x_final_bt, hist_bt = gradient_descent_backtracking(x0, num_iters=10, gamma=0.5, sigma=0.2)
print(f"{'Iter':<6} {'x1':<12} {'x2':<12} {'f(x)':<15} {'α':<10}")
print("-" * 55)
for k, x, fx, alpha in hist_bt:
    alpha_str = f"{alpha:.6f}" if alpha is not None else "N/A"
    print(f"{k:<6} {x[0]:<12.6f} {x[1]:<12.6f} {fx:<15.6f} {alpha_str:<10}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY AFTER 10 ITERATIONS")
print("=" * 60)
print(f"Method 1 (Constant α=0.1):  x = [{x_final_const[0]:.8f}, {x_final_const[1]:.8f}], f(x) = {f(x_final_const):.10f}")
print(f"Method 2 (Exact):           x = [{x_final_exact[0]:.8f}, {x_final_exact[1]:.8f}], f(x) = {f(x_final_exact):.10f}")
print(f"Method 3 (Backtracking):    x = [{x_final_bt[0]:.8f}, {x_final_bt[1]:.8f}], f(x) = {f(x_final_bt):.10f}")
print(f"\nOptimal solution: x* = [0, 0], f(x*) = 0")