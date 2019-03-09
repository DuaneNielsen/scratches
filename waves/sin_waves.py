from matplotlib import pyplot as plt
import numpy as np
from math import pi

x = np.linspace(0,512, 512)
y = np.sin(x * 22 * pi)

alpha = 0.005
p = x * alpha

atten_y = y * np.exp(-p)

z = y * -1
r = np.where(y < 0, z, y)
r = np.log(r)
plt.plot(x,atten_y)

rescale = atten_y * np.exp(p)

plt.plot(x,rescale)

plt.show()