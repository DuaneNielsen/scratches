import imageio
from pathlib import Path
import numpy as np
imageio.plugins.ffmpeg.download()

""""
https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
"""

class ImageVideoReader:
    def __init__(self, file):
        self.reader = imageio.get_reader(file)

    def play(self):
        for i, im in enumerate(self.reader):
            print('mean of frame %i is %1.1f' % (i, im.mean()))

reader = ImageVideoReader(r'c:\data\SpaceInvaders-v4\rl_raw\0.mp4')
reader.play()


"""
write numpy ndarray to png
"""
frame = np.zeros((10,10,3))
name = 'filename'
outputdir = Path('outputdir')
file = outputdir / Path(str(name)).with_suffix('.png')
imageio.imwrite(file, frame)