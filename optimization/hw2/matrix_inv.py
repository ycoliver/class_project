import numpy as np
# matrix = np.array([[1,3,0,0],[2,1,0,0],[1,2,1,0],[3,1,0,1]],dtype=float)

# matrix = np.array([[1,0,0,0],[2,-1,0,0],[1,0,1,0],[3,0,0,1]],dtype=float)
# matrix = np.array([[1,-1,0,0],[2,0,-1,0],[1,0,0,1],[3,0,0,0]],dtype=float)
matrix = np.array([[1,3,-1,0],[2,1,0,-1],[1,2,0,0],[3,1,0,0]],dtype=float)

# matrix = np.array([[1,0,0],[0,2,1],[2,-1,0]],dtype=float)
inv_matrix = np.linalg.inv(matrix)
print(inv_matrix)
# A = np.array([[3],[1],[2],[1]],dtype=float)
# print(np.dot(inv_matrix,A))

