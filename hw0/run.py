import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.interpolate import splprep, splev, interp1d
from math import pi, cos, sin

import data
import config

from robots import Unicycle
from tools import particle_step

NUM_PARTICLES = 200
PATH_RATE = 20
PARTICLE_RATE = 5

# PNOISE = [0.002, 0.002, pi/200]
PNOISE = [0.1, 0.1, pi/16]
PNOISE_less = [0.01, 0.01, pi/100]

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser("Localization Simulation")
    parser.add_argument('dataset', choices=['ds0','ds1'], default=0)
    parser.add_argument('maxsteps', type=int, default=100)
    args = parser.parse_args()

    # ensure correct config values
    config.DATA_SET = int(args.dataset[2])
    import bayes_filter 

    # read data
    file_prefix = "ds{}/ds{}_".format(config.DATA_SET, config.DATA_SET)
    odometry_data = data.read_odometry(file_prefix + 'Odometry.dat')
    measurement_data = data.read_measurements(file_prefix + 'Measurement.dat')
    t_end = odometry_data[args.maxsteps][0]
    
    gt_data = data.read_gt_path(file_prefix + 'Groundtruth.dat')
    gt_interp = interp1d(gt_data[:,0], gt_data[:,1:], axis=0)

    nCommands = len(odometry_data)
    nFeatures = len(measurement_data)
    print "Number of Commands = {}".format(nCommands)
    print "Number of Features = {}".format(nFeatures)

    # initialize some things
    x0 = gt_data[0][1]
    y0 = gt_data[0][2]
    theta0 = gt_data[0][3]

    ##################################################
    # Particle Filter - loops on command events
    ##################################################

    bf = bayes_filter.ParticleFilter(q=[x0, y0, theta0], n=NUM_PARTICLES)
    bf.add_noise(PNOISE)

    dead_reckoned = Unicycle(q=[x0, y0, theta0])
    path_dead_reckoned = [dead_reckoned.get_pose()]

    # path histories for graphs
    gt_path = []
    paths = np.zeros( (NUM_PARTICLES,min(args.maxsteps, nCommands)/PATH_RATE+1,3) )
    for i, p in enumerate(bf.particles):
        paths[i,0,:] = p
    avg_path = [np.average(bf.particles, axis=0)]
    
    # control loop
    loopt0 = time.time()
    ipath = 0
    t_prev = 0
    for ic in range(args.maxsteps):
        t_now = odometry_data[ic][0]
        dt = t_now - t_prev
        cmd_now = odometry_data[ic][1:]

        # Determine features since last loop (measurement)
        fz = []
        while len(measurement_data) > 0 and measurement_data[0].time < t_now:
            fz.append(measurement_data.pop(0))

        # Update Bayes filter and dead reckoned
        bf.update(cmd_now, dt, fz)
        dead_reckoned.control_step(dt)
        dead_reckoned.set_command(cmd_now)

        # Update paths every so often
        if ic % PATH_RATE == 0:
            for i, p in enumerate(bf.particles):
                paths[i,ipath,:] = p
            ipath += 1
        path_dead_reckoned.append(dead_reckoned.get_pose())
        avg_path.append(np.average(bf.particles, axis=0))
        gt_path.append(gt_interp(t_now))

        # Add noise
        bf.add_noise(PNOISE_less)

        t_prev = t_now
        # end control loop

    loopt1 = time.time()
    print "Loop Runtime = {}".format(loopt1 - loopt0)

    ##################################################
    # Drawing
    ##################################################

    print "Plotting ..."

    fig, ax = plt.subplots(figsize=(12, 10))

    # draw intermittent particles
    cmap = plt.cm.cool
    ncolors = np.linspace(0.0, 1.0, len(paths[0]))
    colors = [ cmap(x) for x in ncolors ]
    l_arrow = 0.02
    for i in range(len(paths)):
        xvals = []
        yvals = []
        for j, p in enumerate(paths[i]):
            if j % PARTICLE_RATE== 0:
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
    line_dead_reckoned, = ax.plot(xvals, yvals, color='0.7', linestyle='--', label='Dead Reckoned')

    # draw GT path
    xvals = [x[0] for x in gt_path]
    yvals = [y[1] for y in gt_path]
    line_gt, = ax.plot(xvals, yvals, 'k', label='Ground Truth')

    ax.legend(handles=[line_avg, line_dead_reckoned, line_gt])

    plt.axis('equal')
    plt.plot()
    # plt.savefig('fig_b1.jpg')
    plt.show()
