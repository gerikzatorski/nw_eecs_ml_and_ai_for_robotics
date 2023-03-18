import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d

import config
from tools import read_odometry, read_measurements, read_gt_path
from robots import Unicycle

# Interval at which to capture particles for visualization
PARTICLE_RATE = 300

if __name__ == "__main__":

    # Deterministic runs
    np.random.seed(0)
    
    parser = argparse.ArgumentParser('Localization Simulation')
    parser.add_argument('dataset',
                        help='which  dataset to use',
                        type=str,
                        choices=['ds0','ds1'])
    parser.add_argument('maxsteps',
                        help='maximum number of steps',
                        type=int,
                        default=100)
    parser.add_argument('--plot',
                        help='plot the results? (default: False)',
                        action='store_true')
    args = parser.parse_args()

    # config is used by other modules
    config.DATASET = args.dataset

    # Must import here so that config is shared
    from bayes_filter import ParticleFilter, EKF

    # read data
    file_prefix = f'{config.DATASET}/{config.DATASET}_'
    odometry_data = read_odometry(file_prefix + 'Odometry.dat')
    measurement_data = read_measurements(file_prefix + 'Measurement.dat')

    gt_data = read_gt_path(file_prefix + 'Groundtruth.dat')
    gt_interp = interp1d(gt_data[:,0], gt_data[:,1:], axis=0)
    
    num_commands = len(odometry_data)
    num_features = len(measurement_data)
    num_steps = min(args.maxsteps, num_commands)

    print(f'Running {config.DATASET}')
    print(f'Number of Commands = {num_commands}')
    print(f'Number of Features = {num_features}')
    print(f'Number of Steps    = {num_steps}')
    
    ##################################################
    # Bayes Filters - loops on command events
    ##################################################

    # Start with known pose
    pose0 = gt_data[0][1:4]

    # Setup filters and model
    dead_reckoned = Unicycle(state=pose0)
    pf = ParticleFilter(state=pose0,
                        n=100,
                        step_noise=[0.001, 0.001, np.pi/110],
                        sigmas=[0.2, 0.01, 0.00])
                        # step_noise=[0.001, 0.001, np.pi/64],
                        # sigmas=[0.2, np.pi/32, 0.0])
    ekf = EKF(state=pose0,
              alphas=[0.1, np.pi/16, 0.1, np.pi/16],
              sigmas=[0.4, 0.2, 0.000001],
              outlier_threshold=0.08)
              # alphas=[3.0, np.pi/12, 3.0, np.pi/12],
              # sigmas=[0.6, 0.2, 0.000001],
              # outlier_threshold=0.1)

    # Path histories
    dr_path = []
    gt_path = []
    ekf_path = []
    pf_paths = [[] for i in range(len(pf.particles))]
    pf_avg_path = []

    # Pose error histories
    err_dr = []
    err_pf = []
    err_ekf = []
    tvals = []
    
    # Control loop
    loopt0 = time.time()
    tstart = odometry_data[0][0]
    for ic in range(num_steps - 1):
        # Commands applied from tn to tn+1
        t_now = odometry_data[ic][0]
        dt = odometry_data[ic+1][0] - odometry_data[ic][0]
        cmd_now = odometry_data[ic][1:]
        tvals.append(t_now - tstart)
        
        # Calculate poses
        pose_gt = gt_interp(t_now)
        pose_dr = np.copy(dead_reckoned.state)
        pose_pf = np.average(pf.particles, axis=0)
        pose_ekf = np.copy(ekf.state)

        # Calculate errors
        err_dr.append(np.subtract(pose_gt, pose_dr))
        err_pf.append(np.subtract(pose_gt, pose_pf))
        err_ekf.append(np.subtract(pose_gt, pose_ekf))

        # Updated paths and particles (every so often)
        dr_path.append(pose_dr)
        pf_avg_path.append(pose_pf)
        gt_path.append(pose_gt)
        ekf_path.append(pose_ekf)
        if ic % PARTICLE_RATE == 0:
            for i, p in enumerate(pf.particles):
                pf_paths[i].append(p)

        # Determine features since last loop (measurement)
        fz = []
        while len(measurement_data) > 0 and measurement_data[0][0] < t_now:
            f = measurement_data.pop(0)
            fz.append([f[2], f[3], f[1]])

        # Update paths
        pf.update(cmd_now, dt, fz)
        ekf.update(cmd_now, dt, fz)
        dead_reckoned.set_command(cmd_now)
        dead_reckoned.control_step(dt)
    # End control loop
    
    loopt1 = time.time()
    print(f'Loop Runtime = {loopt1 - loopt0}')
    
    ##################################################
    # Drawing
    ##################################################

    if not args.plot:
        quit()

    print('Plotting ...')

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.axis('equal')
    
    # Draw intermittent particles
    cmap = plt.cm.cool
    ncolors = np.linspace(0.0, 1.0, len(pf_paths[0]))
    colors = [ cmap(x) for x in ncolors ]
    l_arrow = 0.02
    for i in range(len(pf_paths)):
        xvals = []
        yvals = []
        for j, p in enumerate(pf_paths[i]):
            xoffset = l_arrow * np.cos(p[2])
            yoffset = l_arrow * np.sin(p[2])
            ax.arrow(p[0]-xoffset, p[1]-yoffset, xoffset*2, yoffset*2,
                       color=colors[j],
                       alpha=0.8,
                       linewidth=.15,
                       head_width=0.02)
    # Draw all paths
    line_pf_avg, = ax.plot([pose[0] for pose in pf_avg_path],
                        [pose[1] for pose in pf_avg_path],
                        color='0.5', label='Particle Average')
    line_ekf, = ax.plot([pose[0] for pose in ekf_path],
                        [pose[1] for pose in ekf_path],
                        color='y', label='EKF')
    line_dr, = ax.plot([pose[0] for pose in dr_path],
                       [pose[1] for pose in dr_path],
                       color='0.7', linestyle='--', label='Dead Reckoned')
    line_gt, = ax.plot([pose[0] for pose in gt_path],
                       [pose[1] for pose in gt_path],
                       'k', label='Ground Truth')

    ax.legend(handles=[line_pf_avg, line_ekf, line_dr, line_gt])
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'Localized Paths for {config.DATASET}')

    plt.savefig(f'img/paths_{config.DATASET}.png')
    plt.show()

    # Error Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    line_dr_err, = ax.plot(tvals, [np.sqrt(pow(val[0], 2) + pow(val[1], 2)) for val in err_dr],
                           color='0.7', linestyle='--', label='Dead Reckoned')
    line_pf_avg_err, = ax.plot(tvals, [np.sqrt(pow(val[0], 2) + pow(val[1], 2)) for val in err_pf],
                               color='0.5', label='Particle Average')
    line_ekf_err, = ax.plot(tvals, [np.sqrt(pow(val[0], 2) + pow(val[1], 2)) for val in err_ekf],
                            color='y', label='EKF')

    ax.legend(handles=[line_pf_avg_err, line_ekf_err, line_dr_err])
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title(f'Localization Errors for {config.DATASET}')

    plt.savefig(f'img/errors_{config.DATASET}.png')
    plt.show()
