import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import sys
import time

import data
import config

from robots import Robot
from tools import Vector, Pose, particle_step
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

NUM_PARTICLES = 100

if __name__ == "__main__":

    
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
    pose0 = gt[0]
    x0 = pose0.position.x
    y0 = pose0.position.y
    theta0 = pose0.orientation

    ##################################################
    # Particle Filter - loops on command events
    ##################################################

    loopt0 = time.time()

    dead_reckoned = Robot(position=pose0.position, orientation=pose0.orientation)
    path_dead_reckoned = [dead_reckoned.get_pose()]
    
    particles = np.full( (NUM_PARTICLES, 3) , [x0,y0,theta0] )
    weights = np.full( (NUM_PARTICLES,) , 1. / NUM_PARTICLES )
    
    paths = [[] for y in range(NUM_PARTICLES)] 
    for i, p in enumerate(particles): paths[i].append((p[0], p[1])) # todo: replace

    avg_path = []
    
    # control loop
    step = 0
    t_prev = 0
    for ic in range(nCommands):
        t_now = commandQ[ic][0]
        if t_now > t_end: break
        dt = t_now - t_prev

        dead_reckoned.control_step(dt)
        path_dead_reckoned.append(dead_reckoned.get_pose())
        dead_reckoned.set_command(commandQ[ic][1])

        
        # # determine features since last loop (measurement)
        fz = []
        while len(featureQ) > 0 and featureQ[0].time < t_now:
            fz.append(featureQ.pop(0))

        # # given new measurement, run particle filter
        if len(fz) > 0:
            pf.pf_general(particles, weights, commandQ[ic][1], dt, fz)

        for i, p in enumerate(particles):
            particle_step(p, commandQ[ic][1], dt) # control step

        particles = np.random.normal(particles, [0.001, 0.001, pi/64])

        # update paths every so often
        if step % 20 == 0:
            for i, p in enumerate(particles):
                paths[i].append((p[0], p[1]))
            avg_path.append(np.average(particles, axis=0, weights=weights))

        # increment for loop
        t_prev = t_now
        step += 1

    loopt1 = time.time()
    print "Loop Runtime = {}".format(loopt1 - loopt0)
    
    ##################################################
    # Drawing
    ##################################################

    print "Plotting ..."

    fig, ax = plt.subplots(figsize=(12, 10))

    # draw particle paths
    for i, p in enumerate(particles):
        data.draw_tuple_path2(ax, paths[i], color='y')
    data.draw_tuple_path2(ax, avg_path, color='r')
    data.draw_path(ax, path_dead_reckoned, color='b')
    data.draw_path(ax, gt)

    # ax.legend(['ground truth', 'dead reckoned', 'filtered'])
    # ax.get_legend().legendHandles[0].set_color('k')
    # ax.get_legend().legendHandles[1].set_color('b')
    # ax.get_legend().legendHandles[2].set_color('r')

    plt.axis('equal')
    plt.plot()
    # plt.savefig('fig_b1.jpg')
    plt.show()
