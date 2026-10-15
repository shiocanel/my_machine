# Linear systems
# for solving Ax = b

# Gaussian elimination
    # for each column k:
        # find largest entry in column key or below row k
        # swap that row with row k
        # for each i below k:
            # compute multiplier m = A[i][k] / A[k][k]
            # substract m * row k from row i

import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        max_row = k + np.argmax(npabs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[max_row, k]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * ab[[k, k:]]

        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (Ab[i, -1] - Ab[i, i + 1:n] @ x[i+1:n] / Ab[i, i])

        return x