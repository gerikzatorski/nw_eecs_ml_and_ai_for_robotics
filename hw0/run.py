import numpy as np
import matplotlib.pyplot as plt
import sys
import time

import data
import config

from robots import Unicycle
from tools import particle_step
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
NUM_PARTICLES = 200
PATH_RES = 20

PNOISE = [0.001, 0.001, pi/64]

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
    x0 = gt[0][0]
    y0 = gt[0][1]
    theta0 = gt[0][2]

    ##################################################
    # Particle Filter - loops on command events
    ##################################################

    loopt0 = time.time()

    dead_reckoned = Unicycle(q=[x0, y0, theta0])
    path_dead_reckoned = [dead_reckoned.get_pose()]

    particles = np.full( (NUM_PARTICLES, 3) , [x0,y0,theta0] )
    weights = np.full( (NUM_PARTICLES,) , 1. / NUM_PARTICLES )

    particles = np.random.normal(particles, PNOISE)

    
    # sim history for visualizations
    paths = [[] for y in range(NUM_PARTICLES)] 
    for i, p in enumerate(particles): paths[i].append(p)
    avg_path = []
    
    # control loop
    step = 0
    t_prev = 0
    for ic in range(nCommands):
        t_now = commandQ[ic][0]
        if t_now > t_end: break
        dt = t_now - t_prev

        dead_reckoned.control_step(dt)
        dead_reckoned.set_command(commandQ[ic][1])

        # # determine features since last loop (measurement)
        fz = []
        while len(featureQ) > 0 and featureQ[0].time < t_now:
            fz.append(featureQ.pop(0))

        # # given any new features, run particle filter
        if len(fz) > 0:
            pf.pf_general(particles, weights, commandQ[ic][1], dt, fz)

        for i, p in enumerate(particles):
            particle_step(p, commandQ[ic][1], dt) # control step

        # add noise
        particles = np.random.normal(particles, PNOISE)

        # update paths every so often
        if step % PATH_RES == 0:
            for i, p in enumerate(particles):
                paths[i].append(p)
            avg_path.append(np.average(particles, axis=0, weights=weights))
            path_dead_reckoned.append(dead_reckoned.get_pose())

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

    # draw particles
    # for i, p in enumerate(particles):
        # data.draw_particles(ax, paths[i], res=30, cmap=cm)
    cm = plt.cm.cool
    data.draw_particles(ax, paths, res=20, cmap=cm)
    data.draw_path(ax, avg_path, color='#999999')
    data.draw_path(ax, path_dead_reckoned, color='r')
    data.draw_path(ax, gt, color='k')
        
    ax.legend(['ground truth', 'dead reckoned', 'filtered'])
    ax.get_legend().legendHandles[0].set_color('k')
    ax.get_legend().legendHandles[1].set_color('y')
    ax.get_legend().legendHandles[2].set_color('#999999')

    plt.axis('equal')
    plt.plot()
    # plt.savefig('fig_b1.jpg')
    plt.show()
