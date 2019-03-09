import numpy as np

# finds the inverse of a lookup table using the transpose of a permutation matrix

a   = np.array([3,2,0,1])
N   = a.size
rows = np.arange(N)
P   = np.zeros((N,N),dtype=int)
P[rows, a] = 1

print(P)

inv_a = np.where(P.T)[1]
print(inv_a)
