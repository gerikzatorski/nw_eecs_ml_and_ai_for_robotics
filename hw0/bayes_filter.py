import numpy as np
from math import sin, cos, acos, exp, sqrt, atan2, pi

import config
from tools import read_landmark_gt, read_barcodes, norm_pdf, wrap_to_pi

file_prefix = f'{config.DATASET}/{config.DATASET}_'
landmarks = read_landmark_gt(file_prefix + 'Landmark_Groundtruth.dat')
subject_lookup = read_barcodes(file_prefix + 'Barcodes.dat')

class ParticleFilter(object):
    """The basic particle filter, a variant of the Bayes filter based on importance sampling

    Attributes:
        n: number of particles
        particles: a numpy array of all particle poses
        sigmas: sensor model noise parameters
        alphas: motion model noise parameters
    """
    def __init__(self, state=[], n=100, alphas=[], sigmas=[]):
        self.n = n
        self.particles = np.full((self.n, len(state)), state)
        self.sigmas = sigmas
        self.alphas = alphas
        
    def motion_update(self, ut, dt):
        """Particle motion update

        Table 5.6 in Probabilistic Robotics

        Args:
            ut: the commanded control
            dt: the duration to apply the control (s)
        """
        for i, p in enumerate(self.particles):
            dx = ut[0] * dt
            dy = ut[1] * dt
            dtheta = ut[2] * dt

            rot1 = atan2(dy, dx)
            trans = sqrt(pow(dx, 2) + pow(dy, 2))
            rot2 = dtheta - rot1

            rot1hat = np.random.normal(rot1, sqrt(self.alphas[0]*pow(rot1, 2) +
                                                  self.alphas[1]*pow(trans, 2)))
            transhat = np.random.normal(trans, sqrt(self.alphas[2]*pow(trans, 2) +
                                                    self.alphas[3]*pow(rot1, 2) +
                                                    self.alphas[3]*pow(rot2, 2)))
            rot2hat = np.random.normal(rot2, sqrt(self.alphas[0]*pow(rot2, 2) +
                                                  self.alphas[1]*pow(trans, 2)))

            p[0] += transhat * cos(p[2] + rot1hat)
            p[1] += transhat * sin(p[2] + rot1hat)
            p[2] += rot1hat + rot2hat

    def measurement_update(self, zt):
        """Particle measurement update

        Assumes all measurements taken at same time as control step

        Args:
            zt: the measurement data as a list of features
        """
        if len(zt) > 0:
            weights = np.full((self.n,), 1/self.n)
            for i in range(self.n):
                weights[i] = self.importance_factor(zt, self.particles[i])

            # Normalize weights and resample
            weights = np.divide(weights, sum(weights))
            # Copy to prevent sampling from new set
            tmp = np.copy(self.particles)
            for i in range(self.n):
                self.particles[i] = self.particle_sample(tmp, weights)
        
    def update(self, ut, dt, zt):
        """Step forward with the particle filter algorithm

        Args:
            ut: the commanded control
            dt: the time to apply the control
            zt: the measurement data as a list of features
        """
        self.motion_update(ut, dt)
        self.measurement_update(zt)

    def importance_factor(self, fz, pose):
        """Compute the likelihood of a measurement
    
        Args:
            fz: the measurement
            pose: the robot pose

        Returns:
            The probability p( fz | pose, map)
        """
        l = 1.0
        for fi in fz:
            l = l * self.compute_feature_likelihood(fi, pose)
        return l
    
    def compute_feature_likelihood(self, f, pose):
        """Computes the likelihood of a single feature f

        Table 6.4 in Probabilistic Robotics

        Args:
            f: the feature
            pose: the robot pose
    
        Returns:
            The probability p( f | pose, map )
        """
        landmark = landmarks[subject_lookup[f[2]]]

        dx = landmark[0] - pose[0]
        dy = landmark[1] - pose[1]

        # Expected range and bearing measurements
        rhat = sqrt(pow(dx, 2) + pow(dy, 2))
        phihat = atan2(dy, dx) - pose[2]

        # Range and bearing diffs
        dr = f[0] - rhat
        dphi = wrap_to_pi(f[1] - phihat)
        
        return norm_pdf(dr, self.sigmas[0]) * norm_pdf(dphi, self.sigmas[1])

    def particle_sample(self, tmp, weights):
        """Select random particle from tmp
        
        Current implementation faster than np.random.choice(a, p=weights)

        Args:
            tmp: a copy of the filters particles
            weights: indexed probabilities for each particle

        Returns:
            A weighted random sample from the particles
        """
        r = np.random.random_sample() * sum(weights)
        for i, w in enumerate(weights):
            r -= w
            if r < 0:
                break
        return tmp[i]

class EKF(object):
    """The extended Kalman filter localization algorithm.
    
    Table 7.2 in Probabilistic Robotics

    Attributes:
        state: the pose
        alphas: motion model noise parameters
        sigmas: sensor model noise parameters
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
