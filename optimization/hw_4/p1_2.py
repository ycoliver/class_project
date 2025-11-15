import numpy as np
from scipy.optimize import linprog

# Objective function coefficients
c = [-2, -3, -4, -7]  # Maximize z = 2x1 + 3x2 + 4x3 + 7x4 (convert to minimization)

# Coefficients of the equality constraints
A_eq = np.array([[4, 6, -2, 8],
                 [1, 2, -6, 7]])

# Right-hand side of the equality constraints
b_eq = np.array([20, 10])

# Bounds for the variables (x1, x2, x3, x4 >= 0)
x0_bounds = (0, None)
x1_bounds = (0, None)
x2_bounds = (0, None)
x3_bounds = (0, None)

# Function to solve LP relaxation
def solve_lp(c, A_eq, b_eq, bounds):
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    return result

# Solve the LP relaxation (root node)
bounds = [x0_bounds, x1_bounds, x2_bounds, x3_bounds]
result = solve_lp(c, A_eq, b_eq, bounds)

# Print the results of the LP relaxation
print("LP Relaxation Solution:")
print("Objective value (relaxed):", -result.fun)  # Maximize, so negate result
print("Solution (relaxed):", result.x)

# Branch-and-bound procedure
best_known_solution = float('-inf')  # Initialize best-known solution to negative infinity
gap = 1.0  # Set the gap threshold to 1 (termination condition)

def branch_and_bound(c, A_eq, b_eq, bounds, best_known_solution):
    global gap
    
    # First, solve the LP relaxation of the current node
    result = solve_lp(c, A_eq, b_eq, bounds)
    
    if result.success:
        relaxed_solution = result.x
        relaxed_obj = -result.fun  # Maximize, so negate result
        
        print("\nRelaxed Solution at Node:")
        print("Objective:", relaxed_obj)
        print("Solution:", relaxed_solution)
        
        # If all variables are integer, we check if it's the best solution found
        if np.allclose(np.floor(relaxed_solution), relaxed_solution):  # If integer solution
            if relaxed_obj > best_known_solution:
                best_known_solution = relaxed_obj
                print("New best-known solution:", best_known_solution)
        
        # Branching: if not all variables are integer, branch on the first non-integer variable
        if not np.allclose(np.floor(relaxed_solution), relaxed_solution):
            for i, val in enumerate(relaxed_solution):
                if not np.isclose(val, np.floor(val)):  # Branch on this variable
                    lower_bound = np.floor(val)
                    upper_bound = np.ceil(val)
                    
                    # Branch 1: x_i <= lower_bound
                    new_bounds_1 = bounds.copy()
                    new_bounds_1[i] = (0, lower_bound)
                    branch_and_bound(c, A_eq, b_eq, new_bounds_1, best_known_solution)
                    
                    # Branch 2: x_i >= upper_bound
                    new_bounds_2 = bounds.copy()
                    new_bounds_2[i] = (upper_bound, None)
                    branch_and_bound(c, A_eq, b_eq, new_bounds_2, best_known_solution)
        
    return best_known_solution

# Call branch-and-bound method
optimal_solution = branch_and_bound(c, A_eq, b_eq, bounds, best_known_solution)

print("\nOptimal solution found:", optimal_solution)
