import numpy as np
from math import sin, cos, acos, exp, sqrt, atan2, pi

import config
from tools import read_landmark_gt, read_barcodes, norm_pdf, wrap_to_pi

file_prefix = f'{config.DATASET}/{config.DATASET}_'
landmarks = read_landmark_gt(file_prefix + 'Landmark_Groundtruth.dat')
subject_lookup = read_barcodes(file_prefix + 'Barcodes.dat')

##################################################
# Bayes Filter Classes
##################################################

class ParticleFilter(object):
    """The basic particle filter, a variant of the Bayes filter based on importance sampling

    Attributes:
        n: number of particles
        particles: a numpy array of all particle states at any given time
        step_noise: noise applied to particles after each step
        sigmas: sensor model std devs
    """
    def __init__(self, state=[], n=3, step_noise=[], sigmas=[]):
        self.n = n
        self.particles = np.full((self.n, len(state)), state)

        self.step_noise = step_noise
        self.sigmas = sigmas
        
    def update(self, ut, dt, zt):
        """Step forward with the particle filter algorithm

        Args:
            ut: the commanded control
            dt: the time to apply the control
            zt: the measurement data as a list of features
        """
        if len(zt) > 0:
            weights = np.full((self.n,), 1/self.n)
            for i in range(self.n):
                xt = self.particle_preview(self.particles[i], ut, dt)
                weights[i] = self.importance_factor(zt, xt)

            # Normalize weights and resample
            if weights.sum() != 0:
                weights = np.divide(weights, sum(weights))
                # Copy to prevent sampling from new set
                tmp = np.copy(self.particles)
                for i in range(self.n):
                    self.particles[i] = self.particle_sample(tmp, weights)
            
        # Step all particles forward and add noise
        self.particle_step(ut, dt)
        self.add_noise()

    def importance_factor(self, fz, phat):
        """Compute the likelihood of a measurement
    
        Args:
            fz: the measurement
            phat: a pose

        Returns:
            The probability p(f(z) | phat, M )
        """
        l = 1.0
        for fi in fz:
            l = l * self.compute_feature_likelihood(fi, phat)
        return l
    
    def compute_feature_likelihood(self, f, phat):
        """Computes the likelihood of a single feature f
    
        Args:
            f: the feature
            phat: a hypothetical pose [x,y,heading]
    
        Returns:
            The probability p(f | phat, M )
        """
        landmark = landmarks[subject_lookup[f[2]]]

        dx = landmark[0] - phat[0]
        dy = landmark[1] - phat[1]

        # Expected range and bearing measurements
        rhat = sqrt(pow(dx, 2) + pow(dy, 2))
        phihat = atan2(dy, dx) - phat[2]

        # Range and bearing diffs
        dr = f[0] - rhat
        dphi = wrap_to_pi(f[1] - phihat)
        
        return norm_pdf(dr, self.sigmas[0]) * norm_pdf(dphi, self.sigmas[1])

    def add_noise(self):
        """Add noise to all particle poses"""
        self.particles = np.random.normal(self.particles, self.step_noise)
        
    def particle_step(self, ut, dt):
        """Step all particles forward one step

        Args:
            ut: command velocities
            dt: amount of time to apply command
        """
        for p in self.particles:
            p[0] += cos(p[2]) * ut[0] * dt
            p[1] += sin(p[2]) * ut[0] * dt
            p[2] += ut[2] * dt

    def particle_preview(self, p, ut, dt):
        """Preview a particle step

        Args:
            ut: command velocities
            dt: amount of time to apply command
        """
        return [p[0] + cos(p[2]) * ut[0] * dt,
                p[1] + sin(p[2]) * ut[0] * dt,
                p[2] + ut[2] * dt]

    def particle_sample(self, tmp, weights):
        """Select random particle from tmp
        
        Current implementation faster than np.random.choice(a, p=weights)

        Args:
            tmp: a copy of the particles
            weights: particle important factors
        """
        r = np.random.random_sample() * sum(weights)
        for i, w in enumerate(weights):
            r -= w
            if r < 0:
                break
        return tmp[i]

class EKF(object):
    """The extended Kalman filter localization algorithm.
    
    Assumes knowledge of exact correspondences.

    Attributes:
        state: the pose [x, y, theta]
        sigmas: sensor model std devs
        alphas: motion model noise std devs
        covariance: a numpy covariance matrix
        outlier_threshold: measurement probability used to reject outliers
    """
    def __init__(self, state=[], alphas=[], sigmas=[], outlier_threshold=0.002):
        self.state = np.array(state)
        self.sigmas = sigmas
        self.alphas = alphas
        self.covariance = np.zeros((3,3))
        self.outlier_threshold = outlier_threshold

        # Prevent recreation of measurement noise covariance
        self.Qt = np.array([[pow(self.sigmas[0], 2), 0, 0],
                            [0, pow(self.sigmas[1], 2), 0],
                            [0, 0, pow(self.sigmas[2], 2)]])

    def control_update(self, ut, dt):
        """EKF control update (prediction)

        Args:
            ut: the commanded control
            dt: the time to apply the control
        Returns:
            mubar: the predicted mean
            epsilonbar: the predicted covariance
        """
        theta = self.state[2]

        vt = ut[0]
        wt = ut[2]

        # TODO: fix the zero command hack
        if wt == 0.0:
            wt = 0.00001

        # Compute Jacobians Gt and Vt needed to linearize motion model
        Gt = np.array([[1, 0, -vt/wt*cos(theta) + vt/wt*cos(theta + wt*dt)],
                       [0, 1, -vt/wt*sin(theta) + vt/wt*sin(theta + wt*dt)],
                       [0, 0, 1                                                 ]])
        Vt = np.array([[(-sin(theta) + sin(theta + wt*dt)) / wt,  vt*(sin(theta)-sin(theta+wt*dt))/pow(wt, 2) + vt*cos(theta+wt*dt)*dt/wt],
                       [( cos(theta) - cos(theta + wt*dt)) / wt, -vt*(cos(theta)-cos(theta+wt*dt))/pow(wt, 2) + vt*sin(theta+wt*dt)*dt/wt],
                       [0                                            , dt                                                                               ]])

        # Determine the motion noise covariance matrix from the control
        Mt = np.array([[self.alphas[0] * pow(vt, 2) + self.alphas[1] * pow(wt, 2), 0],
                       [0, self.alphas[2] * pow(vt, 2) + self.alphas[3] * pow(wt, 2)]])

        # Calculate predictions
        mubar = np.add(self.state, [-vt/wt*sin(theta) + vt/wt*sin(theta+wt*dt),
                                    vt/wt*cos(theta) - vt/wt*cos(theta+wt*dt),
                                    wt*dt])
        epsilonbar = (Gt @ self.covariance @ Gt.T) + (Vt @ Mt @ Vt.T)

        return mubar, epsilonbar
    
    def measurement_update(self, mubar, epsilonbar, zt):
        """EKF measurement update (correction)

        Args:
            mubar: state mean estimate
            epsilonbar: state covariance estimate
            zt: the measurement data as a list of features
        Returns:
            mu: corrected mean
            epsilon: corrected covariance
            pz: likelihood of the measurement
        """
        pz = 1.0
        for zi in zt:
            # Covariance of additional measurement noise stored as member variable
            # Qt = self.Qt
            
            # Lookup coords of i-th landmark detection and the correct signature
            mjx, mjy = landmarks[subject_lookup[zi[2]]]
            mjs = zi[2]

            # Compute Jacobian H of the measurement model
            q = pow(mjx - mubar[0], 2) + pow(mjy - mubar[1], 2)
            Hi = np.array([[-(mjx-mubar[0])/sqrt(q), -(mjy-mubar[1])/sqrt(q),  0],
                           [ (mjy-mubar[1])/q         , -(mjx-mubar[0])/q         , -1],
                           [ 0                        , 0                         ,  0]])

            # Compute uncertainty corresponding to predicted measurement
            Si = Hi @ epsilonbar @ Hi.T + self.Qt
            
            # Compute Kalman gain
            Ki = epsilonbar @ Hi.T @ np.linalg.pinv(Si)

            # Update estimates for measurements of each feature
            zihat = np.array([ sqrt(q), atan2(mjy - mubar[1], mjx - mubar[0]) - mubar[2], mjs])
            dz = np.add(zi, -zihat)
            dz[1] = wrap_to_pi(dz[1])
            mubar = mubar + (Ki @ dz)
            epsilonbar = (np.identity(len(self.state)) - Ki @ Hi) @ epsilonbar

            # Compute likelihood of a measurement by accumulating products
            pz *= pow(np.linalg.det(2*pi*Si), -1/2) * np.exp(-1/2 * dz.T @ np.linalg.pinv(Si) @ dz)
            
        # Estimates (mubar, epsilonbar) are corrected here (mu, epsilon)
        return mubar, epsilonbar, pz
    
    def update(self, ut, dt, zt):
        """Step forward with the EKF algorithm

        Args:
            ut: the commanded control
            dt: the time to apply the control
            zt: the measurement data as a list of features
        """
        mubar, epsilonbar = self.control_update(ut, dt)
        mu, epsilon, pz = self.measurement_update(mubar, epsilonbar, zt)

        self.covariance = epsilon
        
        # Reject outliers
        if pz < self.outlier_threshold:
            self.state[0] += cos(self.state[2]) * ut[0] * dt
            self.state[1] += sin(self.state[2]) * ut[0] * dt
            self.state[2] += ut[2] * dt
        else:
            self.state = mu
        return
