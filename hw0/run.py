import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import sys
import time

import data
import config

from robots import Robot
from tools import Vector, Pose
from math import pi

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

MAX_RUNTIME = 180

NUM_PARTICLES = 20

if __name__ == "__main__":

    timer0 = time.time()
    
    # parse arguments
    try:
        if int(sys.argv[1]) == 0:
            config.DATA_SET = 0
        elif int(sys.argv[1]) == 1:
            config.DATA_SET = 1
    except:
        config.DATA_SET = 0

    file_prefix = "ds{}/ds{}_".format(config.DATA_SET, config.DATA_SET)

    # import here so that config is updated
    import pf

    # set bounds
    if config.DATA_SET == 0:
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
    gt = data.read_gt_path(file_prefix + 'Groundtruth.dat', tmax=t_end)
    commandQ = data.read_odometry(file_prefix + 'Odometry.dat', tmax=t_end)
    featureQ = data.read_measurements(file_prefix + 'Measurement.dat', tmax=t_end)

    nCommands = len(commandQ)
    nFeatures = len(featureQ)
    print "Number of Commands = {}".format(nCommands)
    print "Number of Features = {}".format(nFeatures)

    # initialize some things
    fig, ax = plt.subplots(figsize=(12, 10))
    pose0 = gt[0]
    t_start = commandQ[0][0]

    ##################################################
    # Dead Reckoned path - loops on command events
    ##################################################

    dead_reckoned = Robot(position=pose0.position, orientation=pose0.orientation)
    path_dead_reckoned = [dead_reckoned.get_pose()]

    t_prev = 0
    step = 0
    for i in range(0, nCommands-1):
        t_now = commandQ[i][0]
        if t_now > t_end: break
        dt = t_now - t_prev
        dead_reckoned.control_step(dt)
        path_dead_reckoned.append(dead_reckoned.get_pose())
        dead_reckoned.set_command(commandQ[i][1])
        t_prev = t_now
        step += 1

    ##################################################
    # Particle Filter - loops on command events
    ##################################################

    particles = np.zeros((NUM_PARTICLES,), dtype=object)
    weights = np.full((NUM_PARTICLES,), 1./NUM_PARTICLES)
    for i in range(NUM_PARTICLES):
        x = Robot(position=pose0.position, orientation=pose0.orientation, noisy=False)
        particles[i] = x

    paths = [[] for y in range(NUM_PARTICLES)] 
    for i, p in enumerate(particles):
        paths[i].append(p.get_pose())

    # control loop
    step = 0
    t_prev = 0
    for ic in range(nCommands):
        t_now = commandQ[ic][0]
        if t_now > t_end: break
        dt = t_now - t_prev

        # determine features since last loop (measurement)
        fz = []
        while len(featureQ) > 0 and featureQ[0].time < t_now:
            fz.append(featureQ.pop(0))

        # given new measurement, run particle filter
        if len(fz) > 0:
            pf.pf_general(particles, weights, dt, fz)
            
        # control step + noise
        for i, p in enumerate(particles):
            p.control_step(dt)
            p.add_noise(0.001,0.001,pi/64)
            paths[i].append(p.get_pose())
            p.set_command(commandQ[ic][1])
        
        # increment for loop
        t_prev = t_now
        step += 1

    
    ##################################################
    # Drawing
    ##################################################

    # draw particle paths
    for i, p in enumerate(particles):
        data.draw_path(ax, paths[i], color='c')
    data.draw_path(ax, path_dead_reckoned, color='b')
    data.draw_path(ax, gt)
    
    timer1 = time.time()
    print "Program runtime = {}".format(timer1 - timer0)

    print "Plotting ..."
    ax.legend(['ground truth', 'dead reckoned', 'filtered'])
    ax.get_legend().legendHandles[0].set_color('k')
    ax.get_legend().legendHandles[1].set_color('b')
    ax.get_legend().legendHandles[2].set_color('r')
    
    plt.axis('equal')
    plt.plot()
    # plt.savefig('fig_b1.jpg')
    plt.show()


