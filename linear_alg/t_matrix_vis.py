import pygame, sys
from pygame.locals import QUIT
import numpy as np
from math import radians, degrees
from numpy import cos, sin, tan, arctan
from numpy.linalg import inv
from collections import OrderedDict
import os

os.environ['SDL_AUDIODRIVER'] = 'dsp'
import math
import time

# Set up the colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
PINK = (255, 51, 255)
MAGENTA = (153, 51, 255)
ORANGE = (255, 128, 0)
YELLOW = (255, 255, 51)

epsilon = np.finfo(float).eps

def to_points(array):
    assert array.shape[0] == 3
    x = array[0:2]
    h = array[2] + epsilon
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
        obj.numpy_points = np.dot(inv(self.t), obj.numpy_points)
        return obj

    def __repr__(self):
        return str(self.t)


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


class Reflect(Transform):
    def __init__(self):
        super().__init__()
        self.t = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])


class Camera(Transform):
    def __init__(self, x, y, theta):
        super().__init__()
        self.t = np.array([[cos(theta), sin(theta), x],
                           [-sin(theta), cos(theta), y],
                           [0, 0, 1]])


class View(Transform):
    def __init__(self, n, f, alpha):
        """
        Transforms to image space
        https://www.youtube.com/watch?v=mpTl003EXCY&t=2279s
        :param n: distance to near clipping plane
        :param f: distance to far clipping plane
        :param alpha: viewable angle
        """
        super().__init__()
        a = (f + n) / (f - n)
        b = 2.0 * f * n / (f - n)
        cot = 1 / tan(alpha / 2)
        self.t = np.array([[a, 0, b],
                           [0, cot, 0],
                           [-1, 0, 0]])


class Shape:
    def __init__(self, color, points, pos=None, scale=None):
        self.color = color
        self._points = points
        self.numpy_points = to_numpy(points)
        if scale:
            Scale(scale, scale)(self)
        if pos:
            Translate(*pos)(self)
        self.colors = None

    def points(self):
        return to_points(self.numpy_points)

    def set_shape(self, numpy):
        h = np.ones((1, numpy.shape[1]))
        self.numpy_points = np.concatenate((numpy, h))

    def draw(self, windowSurface):
        pygame.draw.polygon(windowSurface, self.color, self.points())

    def copy(self):
        return Shape(self.color, self.points())

    def __repr__(self):
        return str(self.numpy_points)

    def __len__(self):
        return self.numpy_points.shape[1]

    def lines(self):
        return LineIter(self)

    def shade(self, light_vector):
        """Light vector is the normal vector of plane light source"""
        lines = LineIter(self)
        n = normal(lines.points, lines.next_points)
        cos_theta = np.absolute(np.dot(n.T, light_vector))
        self.colors = np.expand_dims(cos_theta, axis=1) * np.expand_dims(self.color, axis=0)
        self.colors = np.maximum(self.colors, (np.array(self.color) * 0.3))


def normal(q1, q2):
    """
    unit vector normal to line between q1 and q2
    """
    l = np.linalg.norm(q1 - q2, axis=0)
    n = (q1 - q2) / l
    n = np.roll(n, 1, axis=0)
    n[0] = -n[0]
    return n


class LineIter:
    def __init__(self, shape, homogenous=False):
        self.shape = shape
        self.i = 0
        self.len = len(shape)
        p = self.shape.numpy_points[0:2]
        h = self.shape.numpy_points[2]
        self.points = p / h
        self.next_points = np.roll(self.points, 1, axis=1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i == self.len:
            raise StopIteration
        q1 = self.points[:, self.i]
        q2 = self.next_points[:, self.i]
        self.i += 1
        return q1, q2


class Line:
    def __init__(self, color, start, end):
        self.color = color
        self._points = (start, end)
        self.numpy_points = to_numpy(self._points)

    def points(self):
        return to_points(self.numpy_points)

    def draw(self, windowSurface):
        pygame.draw.line(windowSurface, self.color, self.points()[0], self.points()[1])

    def __repr__(self):
        return str(self.numpy_points)


def build_scene():
    poly = ((125, 100), (375, 100), (375, 300), (125, 300))
    tri = ((125, 100), (375, 100), (375, 300))
    scene = OrderedDict({
        "poly": Shape(RED, poly, pos=(-90, 0), scale=0.25),
        "poly2": Shape(MAGENTA, poly, pos=(120, 70), scale=0.25),
        "poly3": Shape(ORANGE, poly, pos=(250, -80), scale=0.25),
        "tri": Shape(YELLOW, tri, pos=(80, -80), scale=0.25),
        "rear_tri": Shape(BLUE, tri, pos=(-40, -150), scale=0.25)
    })

    return scene


global_translate = Translate(400, 200)


def draw_axis():
    x_axis = Line(BLACK, (-500, 0), (500, 0))
    y_axis = Line(BLACK, (0, -400), (0, 400))
    camera = Shape(BLUE, ((0, 0), (30, -20), (30, 20)))
    fov = Shape(GREEN, ((30, -20), (30, 20), (300, 200), (300, -200)))
    Reflect()(camera)
    Reflect()(fov)

    # draw axis
    global_translate(x_axis)
    global_translate(y_axis)
    global_translate(camera)
    global_translate(fov)
    x_axis.draw(profile)
    y_axis.draw(profile)
    camera.draw(profile)
    fov.draw(profile)


def i_point(q1, q2, d1, d2):
    """
    Find the point that intersects the plane on line defined by q1 qnd q2
    https://www.youtube.com/watch?v=og7hOFypKpQ
    :param q1: point
    :param q2: point
    :param d1: the dot product of q1 with the plane
    :param d1: the dot product of q2 with the plane
    :return: the coordinates of the intermediate point
    """

    t = d1 / (d1 - d2)
    return q1 + t * (q2 - q1)


def fast_clip(shape):
    """
    :param shape:
    :return:
    """
    assert shape.numpy_points.shape[0] == 3
    p = shape.numpy_points[0:2]
    h = shape.numpy_points[2]
    p = p / h
    abs = np.absolute(p)
    max = np.amax(abs, axis=0)
    p_in = np.where(max < 1.0)
    p_out = np.where(max > 1.0)
    return p_in, p_out


class PointList:
    def __init__(self):
        self.q_s = []
        self.d_s = []

    def append(self, q, d=None):
        self.q_s.append(q)
        if d is None:
            p = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
            p = p[:, :, np.newaxis]
            n = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
            n = n[:, np.newaxis, :]
            d = np.matmul(n, q - p).squeeze()
        self.d_s.append(d)

    def numpy(self):
        if len(self.q_s) == 0:
            return np.empty((2, 0)), np.empty((1, 0))
        return np.concatenate(self.q_s).reshape(-1, 2).T, np.array(self.d_s)

    def __repr__(self):
        return str(self.q_s) + '\n' + str(self.d_s)



def clip(shape, p, n):
    assert shape.numpy_points.shape[0] == 3

    """
    v is the vertexs of the polygon
    h is the homogenous co-ordinates (depth)
    p are the points that define the 4 clipping planes
    n are the plane normals pointing inward
    t is the vector q - p
    d is the dot product between the normal and t
    https://www.youtube.com/watch?v=og7hOFypKpQ
    """
    v = shape.numpy_points[0:2]
    h = shape.numpy_points[2] + epsilon
    q = v / h
    p = p[:, :, np.newaxis]
    n = n[:, np.newaxis, :]


    r = q - p
    d = np.matmul(n, r).squeeze(1)

    if np.all(d < 0):
        return False, shape
    else:
        for clip_plane in range(d.shape[0]):
            # todo, compute less dot products
            d = np.matmul(n, q - p).squeeze(1)
            q, d = clip_on_plane(q, d[clip_plane])
            if q.shape[1] == 0:
                return False, shape

        shape.set_shape(q)
        return True, shape


def clip_on_plane(q, d):
    in_p = PointList()
    out_p = PointList()
    poly_size = q.shape[1]
    for i in range(poly_size):
        q1 = q[:, i]
        d1 = d[i]
        next_index = (i + 1) % poly_size
        q2 = q[:, next_index]
        d2 = d[next_index]

        if d1 < 0:  # q1 is out
            out_p.append(q1, d1)
            if d2 >= 0:  # q2 is in
                i_pnt = i_point(q1, q2, d1, d2)
                out_p.append(i_pnt)
                in_p.append(i_pnt)
        else:  # q2 is in
            in_p.append(q1, d1)
            if d2 < 0:  # d2 is out
                i_pnt = i_point(q1, q2, d1, d2)
                out_p.append(i_pnt)
                in_p.append(i_pnt)

    return in_p.numpy()


def update_image_space(image_scene, cam_view):
    clipped_image_space = {}
    p = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    n = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
    for key, shape in image_scene.items():
        cam_view(shape)
        shape_is_in, shape = clip(shape, p, n)
        if shape_is_in:
            clipped_image_space[key] = shape.copy()
    return clipped_image_space


def draw_clipped_image_space(scene):
    draw_scene = copy_scene(scene)
    for key, shape in draw_scene.items():
        Scale(400 / 2, 400 / 2)(shape)
        Translate(400 / 2, 400 / 2)(shape)
        shape.draw(image)


def copy_scene(scene):
    scene_copy = {}
    for key, shape in scene.items():
        scene_copy[key] = shape.copy()
    return scene_copy


def order_points(q1, q2):
    if q1[1] > q2[1]:
        return q1, q2
    else:
        return q2, q1


def project(cam_space, clipped_image_space, resolution):
    dbuffer = np.ones(resolution) * -10.0
    sbuffer = np.zeros((3, resolution))

    for key, shape in clipped_image_space.items():
        for i, (q1, q2) in enumerate(shape.lines()):
            top, bottom = order_points(q1, q2)
            h = math.floor(np.absolute(top[1] - bottom[1]) * resolution // 2)
            z = np.linspace(top[0], bottom[0], h)

            offset = math.floor((top[1] - 1.0) * -resolution // 2)
            slice = np.arange(offset, offset + h)

            index = np.where(z > dbuffer[slice])[0]
            index = index + offset

            color = cam_space[key].colors[i, :]

            sbuffer[:, index] = np.expand_dims(color, axis=1)

            dbuffer[slice] = np.maximum(z, dbuffer[slice])

    return sbuffer


def draw_sbuffer(sbuffer, pixel=(50, 4)):
    w = pixel[0]
    h = pixel[1]
    resolution = sbuffer.shape[1]
    for i in range(resolution):
        pixel = Shape(sbuffer[:, i], ((0, 0), (w, 0), (w, h), (0, h)), pos=(0, (resolution - i - 1) * h))
        pixel.draw(picture)


# Set up pygame
pygame.init()
basicFont = pygame.font.SysFont(None, 48)
pygame.display.set_caption('Camera Transform')

# Set up the window
profile = pygame.Surface((640, 480))
image = pygame.Surface((400, 400))
picture = pygame.Surface((50, 400))
display = pygame.display.set_mode((1280, 760), 0, 32)


def draw(cam_pos, cam_rotation):
    cam_space = build_scene()
    # draw to screen
    profile.fill(WHITE)
    image.fill(WHITE)
    picture.fill(WHITE)
    draw_axis()
    cam_near = 30  # distance of near fov
    cam_far = 300  # distance of far fov
    cam_alpha = arctan(200.0 / 300.0) * 2
    cam_camera = Camera(x=cam_pos[0], y=cam_pos[1], theta=cam_rotation)
    cam_view = View(n=cam_near, f=cam_far, alpha=cam_alpha)

    clipped_cam_space = {}

    for key, shape in cam_space.items():
        cam_camera.inv(shape)
        Reflect()(shape)
        p = np.array([[-cam_near, 0]])
        n = np.array([[-1, 0]])
        in_scene, shape = clip(shape, p, n)
        if in_scene:
            clipped_cam_space[key] = shape

    image_scene = copy_scene(clipped_cam_space)

    for key, shape in cam_space.items():
        global_translate(shape)
        shape.draw(profile)

    clipped_image_space = update_image_space(image_scene, cam_view)
    draw_clipped_image_space(clipped_image_space)
    clipped_cam_space = copy_scene(clipped_image_space)
    for key, shape in clipped_cam_space.items():
        cam_view.inv(shape)
        shape.shade((-1, 0))
    resolution = 100
    sbuffer = project(clipped_cam_space, clipped_image_space, resolution)
    draw_sbuffer(sbuffer)
    display.blit(profile, (0, 0))
    display.blit(image, (670, 50))
    display.blit(picture, (1100, 50))
    # Draw the window onto the screen
    pygame.display.update()


cam_pos = (50, 50)
draw(cam_pos, 0)

frame = 0

# Run the game loop
while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    cam_pos = (50, sin((frame * 0.2)) * 80)
    cam_rotate = sin(frame * 0.1)
    frame += 1
    draw(cam_pos,cam_rotation=cam_rotate)
    #draw((50, 50), cam_rotation=0)
    time.sleep(0.25)
