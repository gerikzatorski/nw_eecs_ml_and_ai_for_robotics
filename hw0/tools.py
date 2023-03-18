import numpy as np

def read_gt_path(filename):
    """Read a robot groundtruth file

    Format is time, x, y, theta

    Returns:
        a numpy array containing all robot pose groundtruths
    """
    return np.loadtxt(filename)

def read_odometry(filename):
    """Read a robot odometry file

    Format is time, linear velocity, angular velocity

    Returns:
        commands: a list of all odometry readings (ie. commands)
    """
    data = np.loadtxt(filename)
    commands = []
    for i, line in enumerate(data):
        commands.append([line[0], line[1], 0, line[2]])
    return commands

def read_measurements(filename):
    """Read a robot measurement file

    Format is time, measured subject #, range, bearing

    Return:
        measurements: a list of all measurements
    """
    other_bots = [5, 14, 41, 32, 23]
    data = np.loadtxt(filename)
    measurement = []
    for line in data:
        # skip measurements of other robots (dynamic obstacles)
        if int(line[1]) in other_bots: continue
        measurement.append([line[0], int(line[1]), line[2], line[3]])
    return measurement

def read_landmark_gt(filename):
    """Read a landmark groundtruth file

    Format is subject #, x, y, x std-dev, y std-dev
    
    Returns:
        landmarks: a mapping of subject numbers to locations
    """
    data = np.loadtxt(filename)
    landmarks = dict()
    for line in data:
        landmarks[int(line[0])] = [line[1], line[2]]
    return landmarks

def read_barcodes(filename):
    """Read a barcodes file
    
    Format is subject #, barcode #

    Returns:
        looktable: a mapping of barcode numbers to subject numbers
    """
    data = np.loadtxt(filename, dtype=int)
    looktable = dict()
    for line in data:
        # looktable[line[0]]= line[1]
        looktable[line[1]]= line[0]
    return looktable

def norm_pdf(v, sigma):
    """The probability density function of a univariate zero-mean normal distribution.

    Args:
        v: the value for which the probability shall be evaluated
        sigma: the standard devation sigma > 0.
    """
    return 1 / np.sqrt(2*np.pi*pow(sigma, 2)) * np.exp(-pow(v, 2) / (2*pow(sigma, 2)))

def wrap_to_pi(theta):
    """Wrap angles to range [-pi, pi] radians"""
    theta = theta % (2 * np.pi)
    if theta > np.pi:
        theta -= 2 * np.pi
    return theta
