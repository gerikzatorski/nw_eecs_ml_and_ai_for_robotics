from robots import Robot
from data import read_odometry, read_measurements, read_landmark_gt, read_gt_path, draw_path, read_barcodes
from tools import Vector, Pose
from pf import pf_general, normalize_weights
from math import pi

import sys
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import copy

# ds0
# range of odometry times    = (1248297556.158, 1248298943.405)
# range of measurement times = (1248297567.247, 1248298943.320)

# ds1
# range of odometry times    = (1288971842.161, 1288973229.039)
# range of measurement times = (1288971842.218, 1288973228.905)

DS0_START = 1248297556.158
DS1_START = 1288971842.218
DS0_END = 1248298943.405
DS1_END = 1288973229.039

TIME_STEP = 0.1
MAX_RUNTIME = None

NUM_PARTICLES = 30

if __name__ == "__main__":
    
    # parse arguments
    dataset = str(sys.argv[1])
    file_prefix = dataset + "/" + dataset + "_"

    # set bounds
    if dataset == "ds0":
        if MAX_RUNTIME is None:
            t_end = DS0_END
        else:
            t_end = DS0_START + MAX_RUNTIME
    else:
        if MAX_RUNTIME is None:
            t_end = DS1_END
        else:
            t_end = DS1_START + MAX_RUNTIME

    # read data
    gt = read_gt_path(file_prefix + 'Groundtruth.dat', tmax=t_end)
    commandQ = read_odometry(file_prefix + 'Odometry.dat', tmax=t_end)
    featureQ = read_measurements(file_prefix + 'Measurement.dat', tmax=t_end)

    nCommands = len(commandQ)
    nFeatures = len(featureQ)

    print "Number of Commands = {}".format(nCommands)
    print "Number of Features = {}".format(nFeatures)

    # initialize some things
    fig, ax = plt.subplots(figsize=(12, 10))
    pose0 = gt[0]
    t_start = commandQ[0][0]

    ##################################################
    # looping on command events (for dead reckoned path)
    ##################################################

    # dead_reckoned = Robot(position=Vector(1.29812900, 1.88315210), orientation=2.82870000)
    tmp_pose = copy.deepcopy(pose0)
    dead_reckoned = Robot(position=tmp_pose.position, orientation=tmp_pose.orientation)
    path_dead_reckoned = [dead_reckoned.get_pose()]

    t_prev = 0
    for i in range(0, nCommands):
        t_now = commandQ[i][0]
        if t_now > t_end: break
        dt = t_now - t_prev
        dead_reckoned.control_step(dt)
        path_dead_reckoned.append(dead_reckoned.get_pose())
        dead_reckoned.set_command(commandQ[i][1])
        t_prev = t_now
    
    ##################################################
    # Particle Filter
    ##################################################

    # generate particles w/ noise
    particles  = []
    weights = []
    w = 1. / NUM_PARTICLES
    for i in range(NUM_PARTICLES):
        tmp_pose = copy.deepcopy(pose0)
        x = Robot(position=tmp_pose.position, orientation=tmp_pose.orientation, noisy=True)        
        particles.append(x)
        weights.append(w)

    ext_paths = []
    for i in range(NUM_PARTICLES): ext_paths.append([])
    for i, p in enumerate(particles):
        ext_paths[i].append(p.get_pose())
    
    # control loop
    t_now = t_start
    t_prev = 0
    ic = 0
    while t_now < t_end and ic < len(commandQ):
        dt = t_now - t_prev

        # determine features since last loop
        fz = []
        while len(featureQ) > 0 and featureQ[0].time < t_now:
            fz.append(featureQ.pop(0))

        # particle filter with new measurements
        if len(fz) > 0:
            # print "We have {} measurements".format(len(fz))
            particles = pf_general(particles, weights, dt, fz)

        # control step + noise
        for p in particles:
            p.control_step(dt)
            # p.add_noise(0.005,0.005,pi/36)
            p.add_noise(0.01,0.01,pi/16)

        # update paths
        for i, p in enumerate(particles):
            ext_paths[i].append(p.get_pose())

        # determine the latest command
        while ic < len(commandQ) and commandQ[ic][0] < t_now:
            ic += 1

        # new commands?
        if ic < len(commandQ):
            for p in particles:
                p.set_command(commandQ[ic][1])

        # loop increments
        t_prev = t_now
        t_now += TIME_STEP

    # draw particle paths
    for i, p in enumerate(particles):
        draw_path(ax, ext_paths[i], color='c')

    # draw dead reckoned and ground truth paths
    draw_path(ax, path_dead_reckoned, color='b')
    draw_path(ax, gt)
        
    print "Plotting ..."
    plt.axis('equal')
    plt.plot()
    # plt.savefig('fig_b1.jpg')
    plt.show()
