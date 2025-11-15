from coptpy import *

env = Envr()
model = env.createModel("CourseSelection")

courses = ["Calculus", "Operations Research", "Data Structures", 
           "Business Statistics", "Computer Simulation", 
           "Intro to Programming", "Forecasting"]

y = {i: model.addVar(vtype=COPT.BINARY, name=f"y_{i}") for i in range(7)}

model.setObjective(quicksum(y[i] for i in range(7)), COPT.MINIMIZE)

model.addConstr(y[0] + y[1] + y[2] + y[3] + y[6] >= 2, name="Math")
model.addConstr(y[1] + y[3] + y[4] + y[6] >= 2, name="OR")
model.addConstr(y[2] + y[4] + y[5] >= 2, name="Computer")

model.addConstr(y[2] <= y[5], name="DS_prereq")
model.addConstr(y[4] <= y[5], name="CS_prereq")
model.addConstr(y[3] <= y[0], name="BS_prereq")
model.addConstr(y[6] <= y[3], name="FC_prereq")

model.solve()

if model.status == COPT.OPTIMAL:
    print(f"最少需要选择 {int(model.objval)} 门课程\n")
    print("选择的课程:")
    for i in range(7):
        if y[i].x > 0.5:
            print(f"  {courses[i]}")