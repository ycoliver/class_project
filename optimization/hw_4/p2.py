import sympy as sp

# Define variables
x1, x2 = sp.symbols('x1 x2')

# Define the objective function f(x)
f = x1**4 + 2*(x1 - x2)*x1**2 + 4*x2**2

# Compute the gradient (partial derivatives with respect to x1 and x2)
grad_f = [sp.diff(f, var) for var in (x1, x2)]

# Solve for the stationary points (set gradient to zero)
stationary_points = sp.solve(grad_f, (x1, x2))

print("Stationary points:", stationary_points)


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function f(x1, x2)
def f(x1, x2):
    return x1**4 + 2*(x1 - x2)*x1**2 + 4*x2**2

# Create a meshgrid for x1 and x2
x1_vals = np.linspace(-2, 2, 100)
x2_vals = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Compute f(x1, x2) for each point in the grid
Z = f(X1, X2)

# Plotting the surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Surface plot of the function f(x1, x2)')

plt.savefig('p2_plot.png')
