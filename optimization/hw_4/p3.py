import numpy as np

# Matrix A1
A1 = np.array([[0, 2, 0], [1, 0, 0], [0, 0, -1]])

# Matrix A2
A2 = np.array([[0, 1, 2], [1, 0, 1], [1, 2, 5]])

# Matrix A3
A3 = np.array([[1, 0, 1], [0, 1, -1], [0, 2, 4]])

# Matrix A4
A4 = np.array([[1, 0, 1], [0, -1, -1], [-1, 1, 0]])


# Compute eigenvalues for each matrix
eig_A1 = np.linalg.eigvals(A1)
eig_A2 = np.linalg.eigvals(A2)
eig_A3 = np.linalg.eigvals(A3)
eig_A4 = np.linalg.eigvals(A4)

print("Eigenvalues of A1:", eig_A1)
print("Eigenvalues of A2:", eig_A2)
print("Eigenvalues of A3:", eig_A3)
print("Eigenvalues of A4:", eig_A4)
