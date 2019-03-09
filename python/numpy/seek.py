import numpy as np
import os

data = np.arange(100, dtype=np.int)
data.tofile("temp")  # save the data

f = open("temp", "rb")  # reopen the file
f.seek(256, os.SEEK_SET)  # seek

x = np.fromfile(f, dtype=np.int)  # read the data into numpy
print(x)