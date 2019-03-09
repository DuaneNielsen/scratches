#!/usr/bin/env python

"""Draw a cube on the screen. every frame we orbit
the camera around by a small amount and it appears
the object is spinning. note i've setup some simple
data structures here to represent a multicolored cube,
we then go through a semi-unoptimized loop to draw
the cube points onto the screen. opengl does all the
hard work for us. :]
"""

import pygame
from pygame.locals import *
import numpy as np
from numpy.linalg import inv, cond
from itertools import combinations
import sys

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
except ImportError:
    print ('The GLCUBE example requires PyOpenGL')
    raise SystemExit



#some simple data for a colored cube
#here we have the 3D point position and color
#for each corner. then we have a list of indices
#that describe each face, and a list of indieces
#that describes each edge


CUBE_POINTS = (
    (0.5, -0.5, -0.5),  (0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5),  (-0.5, -0.5, -0.5),
    (0.5, -0.5, 0.5),   (0.5, 0.5, 0.5),
    (-0.5, -0.5, 0.5),  (-0.5, 0.5, 0.5)
)

#colors are 0-1 floating values
CUBE_COLORS = (
    (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0),
    (1, 0, 1), (1, 1, 1), (0, 0, 1), (0, 1, 1)
)

CUBE_FACES = (
    (0, 1, 2, 3), (3, 2, 7, 6), (6, 7, 5, 4),
    (4, 5, 1, 0), (1, 5, 7, 2), (4, 0, 3, 6)
)

CUBE_EDGES = (
    (0,1), (0,3), (0,4), (2,1), (2,3), (2,7),
    (6,3), (6,4), (6,7), (5,1), (5,4), (5,7),
)


TET_POINTS = (
    (0.0, 0.0, 0.0),  (1.0, 0.0, 0.0),
    (0.5, 1.0, 0.0),  (0.5, 0.5, 1.0)
)

#colors are 0-1 floating values
TET_COLORS = (
    (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)
)

TET_FACES = (
    (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)
)

TET_EDGES = (
    (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
)

SQ_POINTS = (
    (-1.0, -1.0, 0.0),  (1.0, -1.0, 0.0),
    (1.0, 1.0, 0.0),  (-1.0, 1.0, 0.0)
)

#colors are 0-1 floating values
SQ_COLORS = (
    (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0)
)

SQ_FACES = (
    (0, 1, 2, 3),
)

SQ_EDGES = (
    (0,1), (1,2), (2,3), (3,0)
)

TRANS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
)

TRANS_SCALE = np.array(
    [
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
)

TRANS_TRANSLATE = np.array(
    [
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.0, 0.5],
        [0.0, 0.0, 1.0, 0.5],
        [0.0, 0.0, 0.0, 1.0]
    ]
)

from math import degrees
from numpy import sin, cos
theta = degrees(20.0)

TRANS_ROTATE_X = np.array(
    [
        [1.0, 0.0, 0.0, 0.5],
        [0.0, cos(theta), -sin(theta), 0.5],
        [0.0, sin(theta), cos(theta), 0.5],
        [0.0, 0.0, 0.0, 1.0]
    ]
)


def to_points(array):
    assert array.shape[0] == 4
    x = array[0:3]
    h = array[3]
    x = x / h
    return x.T.tolist()


def to_numpy(POINTS):
    x = np.array(POINTS).T
    h = np.ones((1, len(POINTS)))
    x = np.concatenate((x, h))
    return x

def transform(t, POINTS):
    x = to_numpy(POINTS)
    y =  np.dot(t, x)
    return to_points(y)


def recover_transform(x_points, y_points):
    """
    recovers the transformation matrix from 2 sets of 4 points
    :param x_points: the points being transformed
    :param y_points: points after transform is applied
    :return: the 4 x 4 transformation matrix
    """
    assert len(x_points) >= 4 and len(y_points) >= 4
    def find_invertable(points):
        """ find points that form an invertable matrix"""
        for i in combinations(range(points.shape[1]), 4):
            s = points[:, i]
            if cond(s) < 1 / sys.float_info.epsilon:
                return i
        raise Exception

    x = to_numpy(x_points)
    i = find_invertable(x)
    x = to_numpy(x_points)[:, i]
    y = to_numpy(y_points)[:, i]
    t = np.dot(y, inv(x))
    return t



def draw(POINTS, COLORS, FACES, EDGES):
    "draw the cube"
    allpoints = list(zip(POINTS, COLORS))

    glBegin(GL_QUADS)
    for face in FACES:
        for vert in face:
            pos, color = allpoints[vert]
            glColor3fv(color)
            glVertex3fv(pos)
    glEnd()

    glColor3f(1.0, 1.0, 1.0)
    glBegin(GL_LINES)
    for line in EDGES:
        for vert in line:
            pos, color = allpoints[vert]
            glVertex3fv(pos)

    glEnd()

def init_gl_stuff():

    glEnable(GL_DEPTH_TEST)        #use our zbuffer

    #setup the camera
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0,640/480.0,0.1,100.0)    #setup lens
    glTranslatef(0.0, 0.0, -3.0)                #move back
    glRotatef(25, 1, 0, 0)                       #orbit higher

def main():
    "run the demo"
    #initialize pygame and setup an opengl display
    pygame.init()

    fullscreen = False
    pygame.display.set_mode((640,480), OPENGL|DOUBLEBUF)

    init_gl_stuff()

    going = True
    while going:
        #check for quit'n events
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                going = False

            elif event.type == KEYDOWN:
                if event.key == pygame.K_f:
                    if not fullscreen:
                        print("Changing to FULLSCREEN")
                        pygame.display.set_mode((640, 480), OPENGL | DOUBLEBUF | FULLSCREEN)
                    else:
                        print("Changing to windowed mode")
                        pygame.display.set_mode((640, 480), OPENGL | DOUBLEBUF)
                    fullscreen = not fullscreen
                    init_gl_stuff()


        #clear screen and move camera
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        #orbit camera around by 1 degree
        glRotatef(1, 0, 1, 0)

        #draw(TET_POINTS, TET_COLORS, TET_FACES, TET_EDGES)
        y = transform(TRANS_ROTATE_X, CUBE_POINTS)
        t = recover_transform(CUBE_POINTS, y)
        #print(t)
        #draw(y, SQ_COLORS, SQ_FACES, SQ_EDGES)
        draw(y, CUBE_COLORS, CUBE_FACES, CUBE_EDGES)
        pygame.display.flip()
        pygame.time.wait(10)


if __name__ == '__main__': main()