import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from matplotlib.collections import LineCollection
from math import sin, cos

from tools import Feature, Landmark


def draw_particles(axes, paths, res=1, cmap=plt.cm.jet):
    """Draws intermittent arrows for particles"""
    ncolors = np.linspace(0.0, 1.0, len(paths[0]))
    colors = [ cmap(x) for x in ncolors ]
    l = 0.03
    for i in range(len(paths)):
        xvals = []
        yvals = []
        for j, p in enumerate(paths[i]):
            if j % res == 0:
                xoffset = l * cos(p[2])
                yoffset = l * sin(p[2])
                axes.arrow(p[0]-xoffset, p[1]-yoffset, xoffset*2, yoffset*2,
                           color=colors[j],
                           alpha=1.0,
                           linewidth=.1)

def draw_gradient_path(axes, path, cmap=plt.cm.jet):
    """TODO"""
    pass

def draw_path(axes, path, color='k'):
    """Draws a line using a list of tuples represeting position [(x0.y0),(x1,y1),...]"""
    xvals = [x[0] for x in path]
    yvals = [y[1] for y in path]
    axes.plot(xvals, yvals, color=color)

def read_gt_path(filename, tmax=None):
    data = np.loadtxt(filename)
    gt = [] # typles of (timestamp, Pose)
    for line in data:
        if tmax is not None and line[0] > tmax: break
        gt.append([line[1], line[2], line[3]])
    gt = np.array(gt)
    return gt

def read_odometry(filename, tmax=None):
    """Reads from lines with format Time [s], forward velocity [m/s], angular velocity[rad/s]
    Returns: An list of tuples (float time, Twist)
    n"""
    data = np.loadtxt(filename)
    q = []
    for i, line in enumerate(data):
        if tmax is not None and line[0] > tmax: break
        # q.append((line[0], Twist(x=line[1], theta=line[2])))
        q.append([line[0], [line[1], 0, line[2]]])
    # q = np.array(q)
    return q

def read_measurements(filename, tmax=None):
    """Reads from lines with format Time [s], Subject #, range [m], bearing [rad] 
    Returns: An list of tuples (float time, Feature)
    """
    other_bots = [5, 14, 41, 32, 23]
    data = np.loadtxt(filename)
    q = []
    for line in data:
        if tmax is not None and line[0] > tmax: break
        if int(line[1]) in other_bots: continue
        q.append(Feature(time=line[0], barcode=int(line[1]), r=line[2], phi=line[3]))
    # q = np.array(q)
    return q

def read_landmark_gt(filename):
    """Reads from lines with format Subject #, x [m], y [m], x std-dev [m], y std-dev [m]
    Returns: A dict with [Subject #] = [x,y]
    """
    data = np.loadtxt(filename)
    landmarks = dict()
    for line in data:
        landmarks[int(line[0])] = [line[1], line[2]]
    return landmarks

def read_barcodes(filename):
    """Reads from lines with format Subject #, Barcode #
    """
    data = np.loadtxt(filename)
    barcodes = dict()
    for line in data:
        barcodes[int(line[1])] = int(line[0])
    return barcodes
