import numpy as np
from tools import Feature

def read_gt_path(filename):
    return np.loadtxt(filename)

def read_odometry(filename):
    """Reads from lines with format Time [s], forward velocity [m/s], angular velocity[rad/s]
    Returns: Array of commands (time, v_forward, v_sideways, v_angular)
    n"""
    data = np.loadtxt(filename)
    q = []
    for i, line in enumerate(data):
        q.append([line[0], line[1], 0, line[2]])
    q = np.array(q)
    return q

def read_measurements(filename):
    """Reads from lines with format Time [s], Subject #, range [m], bearing [rad] 
    In this dat file, actually means the barcode/signature number
    Returns: An list of tuples (float time, Feature)
    """
    other_bots = [5, 14, 41, 32, 23]
    data = np.loadtxt(filename)
    q = []
    for line in data:
        if int(line[1]) in other_bots: continue
        q.append(Feature(time=line[0], barcode=int(line[1]), r=line[2], phi=line[3]))
    # q = np.array(q)
    return q

def read_features(filename):
    other_bots = [5, 14, 41, 32, 23]
    data = np.loadtxt(filename)
    q = []
    for line in data:
        if int(line[1]) in other_bots: continue
        q.append([line[0], int(line[1]), line[2], line[3]])
    # q = np.array(q)
    return q

    
def read_landmark_gt(filename):
    """Reads from lines with format Subject #, x [m], y [m], x std-dev [m], y std-dev [m]
    Returns: A dict with [Subject #] = [x,y] where subject individually IDs the landmarks
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

def read_landmarks(filename):
    data = np.loadtxt(filename)
    landmarks = dict()
    for line in data:
        # landmarks.append([line[0], line[1], line[2]])
        landmarks[int(line[0])] = [line[1], line[2]]
    # print data[:,:3]
    return landmarks

def read_codes(filename):
    data = np.loadtxt(filename)
    # NumPy version
    # looktable = np.fill((20,), None)
    # for line in data:
    #     np[int(line[0])]= line[1]
    # return looktable
    # dict version
    looktable = dict()
    for line in data:
        looktable[int(line[0])]= line[1]
    return looktable
    
