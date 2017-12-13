import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.interpolate import splprep, splev, interp1d
import scipy.stats
from math import pi, cos, sin, sqrt, atan2

import data
import config

from robots import Unicycle
from tools import particle_step

NUM_PARTICLES = 20
PARTICLE_RATE = 300

PNOISE = [0.1, 0.1, pi/16]

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser("Localization Simulation")
    parser.add_argument('dataset', choices=['ds0','ds1'], default=0)
    parser.add_argument('maxsteps', type=int, default=100)
    parser.add_argument('--plot', default=True)
    args = parser.parse_args()

    # ensure correct config values
    config.DATA_SET = int(args.dataset[2])
    import bayes_filter 

    # read data
    file_prefix = "ds{}/ds{}_".format(config.DATA_SET, config.DATA_SET)
    odometry_data = data.read_odometry(file_prefix + 'Odometry.dat')
    measurement_data = data.read_features(file_prefix + 'Measurement.dat')
    t_end = odometry_data[args.maxsteps][0]

    gt_data = data.read_gt_path(file_prefix + 'Groundtruth.dat')
    gt_interp = interp1d(gt_data[:,0], gt_data[:,1:], axis=0)

    nCommands = len(odometry_data)
    nFeatures = len(measurement_data)
    print "Number of Commands = {}".format(nCommands)
    print "Number of Features = {}".format(nFeatures)

    ##################################################
    # Particle Filter - loops on command events
    ##################################################

    pose0 = gt_data[0][1:4]

    dead_reckoned = Unicycle(q=pose0)
    pf = bayes_filter.ParticleFilter(q=pose0,
                                     n=NUM_PARTICLES,
                                     step_noise=[0.002, 0.002, pi/100],
                                     sigmas=[2.0, pi/32, 0.0])
    # pf.add_noise(PNOISE)

    ekf = bayes_filter.EKF(state=pose0,
                           alphas=[0.1, 0.1, 5*pi, pi/3],
                           sigmas=[0.0721, 0.0494, 0.000000001])

    # path histories for graphs
    path_dead_reckoned = []
    gt_path = []
    ekf_path = []
    avg_path = []
    paths = np.zeros( (NUM_PARTICLES,min(args.maxsteps, nCommands)/PARTICLE_RATE+1,3) )
    for i, p in enumerate(pf.particles):
        paths[i,0,:] = p

    err_fz = [] # real measurements errors
    err_dr = []
    err_pf = []
    err_ekf = []
    
    # control loop
    loopt0 = time.time()
    ipath = 0
    t_prev = 0
    for ic in range(min(args.maxsteps, nCommands-1)):
        
        t_now = odometry_data[ic][0]
        dt = t_now - t_prev
        cmd_now = odometry_data[ic][1:]

        # Determine features since last loop (measurement)
        fz = []
        while len(measurement_data) > 0 and measurement_data[0][0] < t_now:
            f = measurement_data.pop(0)
            fz.append([f[2], f[3], f[1]])

        # Update Bayes filters and dead reckoned
        pf.update(cmd_now, dt, fz)
        ekf.update(cmd_now, dt, fz)
        dead_reckoned.control_step(dt)
        dead_reckoned.set_command(cmd_now)

        # Calculate poses
        pose_gt = gt_interp(t_now)
        pose_dr = dead_reckoned.get_pose()
        pose_pf = np.average(pf.particles, axis=0)
        pose_ekf = ekf.state

        # Update paths and particles (every so often)
        path_dead_reckoned.append(pose_dr)
        avg_path.append(pose_pf)
        gt_path.append(pose_gt)
        ekf_path.append(pose_ekf)
        if ic % PARTICLE_RATE == 0:
            for i, p in enumerate(pf.particles):
                paths[i,ipath,:] = p
            ipath += 1

        err_dr.append(np.add(pose_gt, -pose_dr))
        err_pf.append(np.add(pose_gt, -pose_pf))
        err_ekf.append(np.add(pose_gt, -pose_ekf))
        
        # for zi in fz:
        #     lm = bayes_filter.M[bayes_filter.c(zi[2])]
        #     rr = sqrt(pow(pose_gt[0] - lm[0], 2) + pow(pose_gt[1] - lm[1], 2))
        #     rphi = atan2(lm[1] - pose_gt[1], lm[0] - pose_gt[0]) - pose_gt[2]
        #     mr = f[2]
        #     mphi = f[3]
        #     dr = rr - mr
        #     dphi = rphi - mphi
        #     err_fz.append([dr, dphi])
        #     # print "Real Measurement Error = {}".format([dr, dphi])

        t_prev = t_now
        # end control loop

    loopt1 = time.time()
    print "Loop Runtime = {}".format(loopt1 - loopt0)

    ##################################################
    # Drawing
    ##################################################

    # if plot
    print "Plotting ..."

    fig, ax = plt.subplots(2, figsize=(12, 10))

    # draw intermittent particles
    cmap = plt.cm.cool
    ncolors = np.linspace(0.0, 1.0, len(paths[0]))
    colors = [ cmap(x) for x in ncolors ]
    l_arrow = 0.02
    for i in range(len(paths)):
        xvals = []
        yvals = []
        for j, p in enumerate(paths[i]):
            xoffset = l_arrow * cos(p[2])
            yoffset = l_arrow * sin(p[2])
            ax[0].arrow(p[0]-xoffset, p[1]-yoffset, xoffset*2, yoffset*2,
                       color=colors[j],
                       alpha=0.8,
                       linewidth=.15,
                       head_width=0.02)
    
    # draw avg particle path (not smoothed)
    xvals = [pose[0] for pose in avg_path]
    yvals = [pose[1] for pose in avg_path]
    line_avg, = ax[0].plot(xvals, yvals, color='0.5', label='Particle Avg (discrete)')

    # draw smoothed avg particle path
    # xvals = [pose[0] for pose in avg_path]
    # yvals = [pose[1] for pose in avg_path]
    # tck, u = splprep([xvals, yvals], s=0.0)
    # unew = np.arange(0, 1.01, 0.01)
    # out = splev(unew, tck)
    # line_avg_smooth, = ax[0].plot(out[0], out[1], color='0.5', label='Particle Average')

    # draw ekf path
    xvals = [pose[0] for pose in ekf_path]
    yvals = [pose[1] for pose in ekf_path]
    line_ekf, = ax[0].plot(xvals, yvals, color='y', label='EKF')
    
    # draw dead reckoned path
    xvals = [pose[0] for pose in path_dead_reckoned]
    yvals = [pose[1] for pose in path_dead_reckoned]
    line_dead_reckoned, = ax[0].plot(xvals, yvals, color='0.7', linestyle='--', label='Dead Reckoned')

    # draw GT path
    xvals = [pose[0] for pose in gt_path]
    yvals = [pose[1] for pose in gt_path]
    line_gt, = ax[0].plot(xvals, yvals, 'k', label='Ground Truth')
    
    # ax[0].legend(handles=[line_avg, line_dead_reckoned, line_gt])

    # Error Subplot
    ax[1].plot([sqrt(pow(val[0], 2) + pow(val[1], 2)) for val in err_dr], color='0.7')
    ax[1].plot([sqrt(pow(val[0], 2) + pow(val[1], 2)) for val in err_pf], 'b-')
    ax[1].plot([sqrt(pow(val[0], 2) + pow(val[1], 2)) for val in err_ekf], 'y-')

    # print scipy.stats.describe(err_fz[0])
    # print scipy.stats.describe(err_fz[1])
    # import sys; sys.exit()

    # plt.axis('equal')
    plt.plot()
    # plt.savefig('img/fig_ds0.png')
    plt.show()
