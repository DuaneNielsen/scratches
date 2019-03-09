import pygame, sys
from pygame.locals import *
import numpy as np
from math import radians, degrees
from numpy import cos, sin
from numpy.linalg import inv
from collections import OrderedDict

# Set up pygame
pygame.init()

# Set up the window
windowSurface = pygame.display.set_mode((640, 480), 0, 32)
pygame.display.set_caption('Camera Transform')

# Set up the colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)

# Set up fonts
basicFont = pygame.font.SysFont(None, 48)


def to_points(array):
    assert array.shape[0] == 3
    x = array[0:2]
    h = array[2]
    x = x / h
    return x.T.tolist()


def to_numpy(POINTS):
    x = np.array(POINTS).T
    h = np.ones((1, len(POINTS)))
    x = np.concatenate((x, h))
    return x


def transform(t, POINTS):
    x = to_numpy(POINTS)
    y = np.dot(t, x)
    return to_points(y)


class Transform:
    def __init__(self):
        self.t = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])

    def apply(self, obj):
        return np.dot(self.t, obj)

    def __call__(self, obj):
        obj.numpy_points = np.dot(self.t, obj.numpy_points)
        return obj

    def inv(self, obj):
        obj.numpy_points =  np.dot(inv(self.t), obj.numpy_points)
        return obj


class Translate(Transform):
    def __init__(self, x, y):
        super().__init__()
        self.t = np.array([[1, 0, x],
                           [0, 1, y],
                           [0, 0, 1]])


class Scale(Transform):
    def __init__(self, w, h):
        super().__init__()
        self.t = np.array([[w, 0, 0],
                           [0, h, 0],
                           [0, 0, 1]])


class Rotate(Transform):
    def __init__(self, theta):
        super().__init__()
        self.t = np.array([[cos(theta), sin(theta), 0],
                           [-sin(theta), cos(theta), 0],
                           [0, 0, 1]])

class Shape:
    def __init__(self, color, points):
        self.color = color
        self._points = points
        self.numpy_points = to_numpy(points)

    def points(self):
        return to_points(self.numpy_points)

    def draw(self, windowSurface):
        pygame.draw.polygon(windowSurface, self.color, self.points())

class Line:
    def __init__(self, color, start, end):
        self.color = color
        self._points = (start, end)
        self.numpy_points = to_numpy(self._points)

    def points(self):
        return to_points(self.numpy_points)

    def draw(self, windowSurface):
        pygame.draw.line(windowSurface, self.color, self.points()[0], self.points()[1])

# Draw the white background onto the surface
windowSurface.fill(WHITE)

camera = ((0, 0), (30, -20), (30, 20))
fov = ((30, -20), (30, 20), (300, 200), (300, -200))


# Draw a blue polygon onto the surface
# pygame.draw.polygon(windowSurface, BLUE, ((250, 0), (500,200),(250,400), (0,200) ))

poly = ((125, 100), (375, 100), (375, 300), (125, 300))

scene = OrderedDict({"fov": Shape(RED, fov), "poly": Shape(GREEN, poly), "camera": Shape(BLUE, camera)})
x_axis = Line(BLACK, (-500,0), (500, 0))
y_axis = Line(BLACK, (0,-400), (0, 400))

camera_pos = 50, 50
camera_vector = radians(0.0)

cam_trans = Translate(camera_pos[0], camera_pos[1])
cam_rotate = Rotate(camera_vector)

Scale(0.5, 0.5)(scene["poly"])
Translate(120, -40)(scene["poly"])

cam_trans.inv(scene["poly"])
cam_rotate.inv(scene["poly"])


global_translate = Translate(100,200)

for key, shape in scene.items():
    global_translate(shape)
    shape.draw(windowSurface)

global_translate(x_axis)
global_translate(y_axis)

x_axis.draw(windowSurface)
y_axis.draw(windowSurface)

# Draw a red circle onto the surface
# pygame.draw.circle(windowSurface, RED, (250,200), 125)

# Get a pixel array of the surface
pixArray = pygame.PixelArray(windowSurface)
pixArray[480][380] = BLACK
del pixArray

# Draw the text onto the surface
# windowSurface.blit(text,textRect)

# Draw the window onto the screen
pygame.display.update()

# Run the game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
