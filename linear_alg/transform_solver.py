import numpy as np
from numpy.linalg import inv

triagle = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 1, 1],
]

x = np.array(triagle, dtype=float)

i = [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]

scale = [[2, 0, 0],
        [0, 2, 0],
        [0, 0, 1]]


test_transform = np.array(scale, dtype=float)

y = np.dot(test_transform, x)

inv_x = inv(x)
recovered_transform = np.dot(y, inv_x)

y_prime = np.dot(recovered_transform, x)


print(test_transform)
print(recovered_transform)

print(y)
print(y_prime)
