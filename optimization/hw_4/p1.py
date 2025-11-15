import coptpy as cp
from coptpy import COPT

def branch_and_bound():
    # Initialize environment and model
    env = cp.Envr()
    model = env.createModel("MILP_BnB")

    # Decision variables
    x1 = model.addVar(vtype=COPT.INTEGER, name="x1")
    x2 = model.addVar(vtype=COPT.INTEGER, name="x2")
    x3 = model.addVar(vtype=COPT.INTEGER, name="x3")
    x4 = model.addVar(vtype=COPT.INTEGER, name="x4")
    
    # Objective function
    model.setObjective(2*x1 + 3*x2 + 4*x3 + 7*x4, COPT.MAXIMIZE)
    
    # Constraints
    model.addConstr(4*x1 + 6*x2 - 2*x3 + 8*x4 == 20, name="constraint_1")
    model.addConstr(x1 + 2*x2 - 6*x3 + 7*x4 == 10, name="constraint_2")
    
    # Solve the LP relaxation (continuous relaxation of integer variables)
    model.solve()
    
    if model.status == COPT.OPTIMAL:
        print(f"Root LP solution: {model.objval}")
        print(f"Solution: x1={x1.x}, x2={x2.x}, x3={x3.x}, x4={x4.x}")
        
        # Check if all variables are integers, if so, we have an optimal solution
        if all(var.x == int(var.x) for var in [x1, x2, x3, x4]):
            print("Optimal solution found!")
        else:
            print("Branching required.")
            # Perform branching here
            fractional_var = min([x1, x2, x3, x4], key=lambda var: var.x % 1)
            fractional_value = fractional_var.x
            
            # Branching constraint: create two subproblems, one with x_i <= floor(x_i*) and one with x_i >= ceil(x_i*)
            lower_bound = int(fractional_value)
            upper_bound = int(fractional_value) + 1
            
            # Create two child models with different constraints
            # Subproblem 1: x_i <= lower_bound
            submodel_1 = env.createModel("Subproblem_1")
            submodel_1.addVar(vtype=COPT.INTEGER, name="x1")
            submodel_1.addVar(vtype=COPT.INTEGER, name="x2")
            submodel_1.addVar(vtype=COPT.INTEGER, name="x3")
            submodel_1.addVar(vtype=COPT.INTEGER, name="x4")
            submodel_1.addConstr(4*x1 + 6*x2 - 2*x3 + 8*x4 == 20)
            submodel_1.addConstr(x1 + 2*x2 - 6*x3 + 7*x4 == 10)
            submodel_1.addConstr(fractional_var <= lower_bound)

            submodel_1.solve()
            if submodel_1.status == COPT.OPTIMAL:
                print("Subproblem 1 optimal: ", submodel_1.objval)

            # Subproblem 2: x_i >= upper_bound
            submodel_2 = env.createModel("Subproblem_2")
            submodel_2.addVar(vtype=COPT.INTEGER, name="x1")
            submodel_2.addVar(vtype=COPT.INTEGER, name="x2")
            submodel_2.addVar(vtype=COPT.INTEGER, name="x3")
            submodel_2.addVar(vtype=COPT.INTEGER, name="x4")
            submodel_2.addConstr(4*x1 + 6*x2 - 2*x3 + 8*x4 == 20)
            submodel_2.addConstr(x1 + 2*x2 - 6*x3 + 7*x4 == 10)
            submodel_2.addConstr(fractional_var >= upper_bound)

            submodel_2.solve()
            if submodel_2.status == COPT.OPTIMAL:
                print("Subproblem 2 optimal: ", submodel_2.objval)

if __name__ == "__main__":
    branch_and_bound()
