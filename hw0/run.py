import numpy as np
import matplotlib.pyplot as plt
import sys
import time

from scipy.interpolate import splprep, splev
from math import pi, cos, sin

import data
import config

from robots import Unicycle
from tools import particle_step, particle_step_np

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

MAX_RUNTIME = 80
NUM_PARTICLES = 100
PNOISE = [0.002, 0.002, pi/200]
# PNOISE = [0.003, 0.003, pi/16]

PATH_RATE = 15

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
    x0 = gt[0][1]
    y0 = gt[0][2]
    theta0 = gt[0][3]

    ##################################################
    # Particle Filter - loops on command events
    ##################################################

    loopt0 = time.time()

    dead_reckoned = Unicycle(q=[x0, y0, theta0])
    path_dead_reckoned = [dead_reckoned.get_pose()]

    particles = np.full( (NUM_PARTICLES, 3) , [x0,y0,theta0] )
    weights = np.full( (NUM_PARTICLES,) , 1. / NUM_PARTICLES )

    # some initial noise
    # particles = np.random.normal(particles, PNOISE)

    paths = np.zeros( (NUM_PARTICLES,nCommands/PATH_RATE+1,3) )
    for i, p in enumerate(particles):
        paths[i,0,:] = p
    avg_path = [np.average(particles, axis=0, weights=weights)]
    pftimes = []
    
    # control loop
    step = 0
    ipath = 0
    t_prev = 0
    for ic in range(nCommands):
        t_now = commandQ[ic][0]
        cmd_now = commandQ[ic][1:]
        if t_now > t_end: break
        dt = t_now - t_prev

        dead_reckoned.control_step(dt)
        dead_reckoned.set_command(cmd_now)

        # # determine features since last loop (measurement)
        fz = []
        while len(featureQ) > 0 and featureQ[0].time < t_now:
            fz.append(featureQ.pop(0))

        # given any new features, run particle filter
        if len(fz) > 0:
            weights = np.full( (NUM_PARTICLES,) , 1. / NUM_PARTICLES )
            pf.pf_general(particles, weights, cmd_now, dt, fz)
            pftimes.append(t_now)

        for p in particles:
            particle_step(p, cmd_now, dt)

        # update paths every so often
        if step % PATH_RATE == 0:
            for i, p in enumerate(particles):
                paths[i,ipath,:] = p
            ipath += 1
        path_dead_reckoned.append(dead_reckoned.get_pose())
        avg_path.append(np.average(particles, axis=0))

        # add noise
        particles = np.random.normal(particles, PNOISE)

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

    # Graph measurment times along GT path
    # measurement_pts = []
    # igt = 0
    # for t in pftimes:
    #     while t > gt[igt][0]:
    #         igt += 1
    #     pt = gt
    #     measurement_pts.append(gt[igt,1:])
    # xvals = [x[0] for x in measurement_pts]
    # yvals = [y[1] for y in measurement_pts]
    # ax.scatter(xvals, yvals, color='k', alpha=alpha)

    # draw intermittent particles
    res = 20
    cmap = plt.cm.cool
    ncolors = np.linspace(0.0, 1.0, len(paths[0]))
    colors = [ cmap(x) for x in ncolors ]
    l_arrow = 0.02
    for i in range(len(paths)):
        xvals = []
        yvals = []
        for j, p in enumerate(paths[i]):
            if j % res == 0:
                xoffset = l_arrow * cos(p[2])
                yoffset = l_arrow * sin(p[2])
                ax.arrow(p[0]-xoffset, p[1]-yoffset, xoffset*2, yoffset*2,
                           color=colors[j],
                           alpha=0.8,
                           linewidth=.15,
                           head_width=0.02)

    # draw avg path (not smoothed)
    xvals = [x[0] for x in avg_path]
    yvals = [y[1] for y in avg_path]
    line_avg, = ax.plot(xvals, yvals, color='0.5', label='Particle Avg (discrete)')

    # draw smoothed avg particle path
    # xvals = [x[0] for x in avg_path]
    # yvals = [y[1] for y in avg_path]
    # tck, u = splprep([xvals, yvals], s=0.0)
    # unew = np.arange(0, 1.01, 0.01)
    # out = splev(unew, tck)
    # line_avg_smooth, = ax.plot(out[0], out[1], color='0.5', label='Particle Average')
                
    # draw dead reckoned path
    xvals = [x[0] for x in path_dead_reckoned]
    yvals = [y[1] for y in path_dead_reckoned]
    line_dead_reckoned, = ax.plot(xvals, yvals, color='0.9', linestyle='--', label='Dead Reckoned')

    # draw GT path
    xvals = [x[0] for x in gt[:,1:]]
    yvals = [y[1] for y in gt[:,1:]]
    line_gt, = ax.plot(xvals, yvals, 'k', label='Ground Truth')

    # ax.legend(handles=[line_avg_smooth, line_dead_reckoned, line_gt])

    plt.axis('equal')
    plt.plot()
    # plt.savefig('fig_b1.jpg')
    plt.show()
