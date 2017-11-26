import numpy as np
from tools import Vector, Pose, Twist, Feature, Landmark

import matplotlib.pyplot as plt
import matplotlib.lines as lines

def draw_path(axes, path, res=20, color='k'):
    n = len(path)
    pose_prev = path[0]
    for i in range(0, len(path), res):
        axes.add_line(lines.Line2D([pose_prev.position.x, path[i].position.x],
                                   [pose_prev.position.y, path[i].position.y],
                                   color=color))
        pose_prev = path[i]

def read_gt_path(filename, tmax=None):
    data = np.loadtxt(filename)
    gt = [] # typles of (timestamp, Pose)
    for line in data:
        if tmax is not None and line[0] > tmax: break
        gt.append(Pose(line[1], line[2], line[3]))
    return gt

def read_odometry(filename, tmax=None):
    """Reads from lines with format Time [s], forward velocity [m/s], angular velocity[rad/s]
    Returns: An array of tuples (float time, Twist)
    n"""
    data = np.loadtxt(filename)
    q = []
    for i, line in enumerate(data):
        if tmax is not None and line[0] > tmax: break
        q.append((line[0], Twist(x=line[1], theta=line[2])))
    return q

def read_measurements(filename, tmax=None):
    """Reads from lines with format Time [s], Subject #, range [m], bearing [rad] 
    Returns: An array of tuples (float time, Feature)
    """
    other_bots = [5, 14, 41, 32, 23]
    data = np.loadtxt(filename)
    q = []
    for line in data:
        if tmax is not None and line[0] > tmax: break
        if int(line[1]) in other_bots: continue
        q.append(Feature(time=line[0], barcode=int(line[1]), r=line[2], phi=line[3]))
    return q

def read_landmark_gt(filename):
    """Reads from lines with format Subject #, x [m], y [m], x std-dev [m], y std-dev [m]
    Returns: A dict with [Subject #] = Vector(x,y)
    """
    data = np.loadtxt(filename)
    landmarks = dict()
    for line in data:
        landmarks[int(line[0])] = Vector(line[1], line[2])
        # landmarks.append(Landmark(subject=int(line[0]), x=line[1], y=line[2]))
    return landmarks

def read_barcodes(filename):
    """Reads from lines with format Subject #, Barcode #
    """
    data = np.loadtxt(filename)
    barcodes = dict()
    for line in data:
        barcodes[int(line[1])] = int(line[0])
    return barcodes
