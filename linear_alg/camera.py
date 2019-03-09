from math import degrees
import numpy as np

cam_dir = ()
cam_pos = (0, 0, 0)
x, y, z = cam_pos

n = 1.0
f = 10.0
alpha = degrees(45)

view = np.array([
    [0,  0, 1, x],
    [0, -1, 0, y],
    [1, 0,  0, z],
    [0, 0,  0, 1]
])

# camera is centered on origin, facing in -z

camera = np.array([
    [1/np.tan(alpha/2.0), 0, 0,                0],
    [0, 1/np.tan(alpha/2.0), 0,                0],
    [0, 0,                   (f + n)/(f - n), -1],
    [0, 0,                   2*f*n/(f- n),     0]
])