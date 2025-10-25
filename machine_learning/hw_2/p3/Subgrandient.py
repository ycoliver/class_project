import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

A = np.load('A.npy')
b = np.load('b.npy')
x_star = np.load('x_star.npy')

n, d = A.shape

def subgradient_sign(z):
    signs = np.zeros_like(z)
    signs[z > 0] = 1
    signs[z < 0] = -1
    signs[z == 0] = np.random.choice([-1, 1], size=np.sum(z == 0))
    return signs

def compute_subgradient(A, b, x):
    residual = A @ x - b
    signs = subgradient_sign(residual)
    return A.T @ signs

def subgradient_method(A, b, x_star, learning_rate_fn, max_iter=300):
    x = np.zeros(d)
    gaps = []
    
    for k in range(max_iter):
        gap = np.linalg.norm(x - x_star)
        gaps.append(gap)
        
        g = compute_subgradient(A, b, x)
        mu = learning_rate_fn(k)
        x = x - mu * g
    
    return gaps

learning_rate_configs = {
    'Constant (μ=0.005)': lambda k: 0.005,
    'Polynomial (μ=1/√k)': lambda k: 1.0 / np.sqrt(k + 1),
    'Geometric (μ=0.01×0.9^k)': lambda k: 0.01 * (0.9 ** k)
}

results = {}
for name, lr_fn in learning_rate_configs.items():
    print(f"Running subgradient method with {name}...")
    gaps = subgradient_method(A, b, x_star, lr_fn, max_iter=300)
    results[name] = gaps

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, gaps in results.items():
    axes[0].plot(gaps, label=name, linewidth=2)
    axes[1].semilogy(gaps, label=name, linewidth=2)

axes[0].set_xlabel('Iteration k', fontsize=12)
axes[0].set_ylabel('Optimality Gap ||x_k - x*||_2', fontsize=12)
axes[0].set_title('Subgradient Method Convergence (Normal Scale)', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Iteration k', fontsize=12)
axes[1].set_ylabel('Optimality Gap ||x_k - x*||_2 (log scale)', fontsize=12)
axes[1].set_title('Subgradient Method Convergence (Log Scale)', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('subgradient_convergence.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nFinal optimality gaps:")
for name, gaps in results.items():
    print(f"{name}: {gaps[-1]:.6f}")