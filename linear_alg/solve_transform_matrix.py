import numpy as np
from numpy.linalg import inv

x = np.array([[2, 6,  8],
             [2, -2, 6],
             [1, 1,  1]])

y = np.array([[-2, -10, -14],
              [-2,  -4,  10],
              [ 1,   1,   1]])

# t x = y
# t x inv(x) = y inv(x)
# t I = y inv(x)
# t = y inv(x)

t = np.dot(y, inv(x))

print(t)

y = np.dot(t, x)

print(y)