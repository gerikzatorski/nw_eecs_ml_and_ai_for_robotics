import numpy as np

from math import cos, sin, acos, exp, sqrt, atan2, pi

# Reading Data using config files
import config
from data import read_landmark_gt, read_barcodes, read_landmarks, read_codes
file_prefix = "ds{}/ds{}_".format(config.DATA_SET, config.DATA_SET)
landmarks = read_landmark_gt(file_prefix + 'Landmark_Groundtruth.dat')
subject_lookup = read_barcodes(file_prefix + 'Barcodes.dat')

M = read_landmarks(file_prefix + 'Landmark_Groundtruth.dat')
looktable = read_codes(file_prefix + 'Barcodes.dat') # keys are unique id, values are the feature type

def c(feature):
    return looktable.keys()[looktable.values().index(feature)]

##################################################
# Bayes Filter Classes
##################################################

class ParticleFilter(object):
    """The basic particle filter, a variant of the Bayes filter based on importance sampling

    This is the algorithm in Probabilistic Robotics

    Attributes:
        q: particle state vector
        n: number of particles
        particles: a numpy array of all particle states at any given time

    """
    def __init__(self, q=[], n=3):
        self.q = q
        self.n = n
        self.particles = np.full( (self.n, len(self.q)) , self.q )
        
    def update(self, ut, dt, zt):
        """Step forward with the particle filter algorithm

        Args:
            ut: the commanded control
            dt: the time to apply the control
            zt: the measurement data as a list of features
        """
        if len(zt) > 0:
            weights = np.full( (self.n,) , 1. / self.n )
            for i in range(self.n):
                xt = self.particle_preview(self.particles[i], ut, dt)
                weights[i] = importance_factor(zt, xt)

            # normalize weights
            weights = np.divide(weights, sum(weights))

            # if degeneracy is too high, resample
            tmp = np.copy(self.particles)
            if 0.5 > 1/np.square(weights).sum() / self.n:
                for i in range(self.n):
                    psample = self.particle_sample(tmp, weights)
                    self.particles[i] = psample
                # self.add_noise([0.00001, 0.00001, pi/400])

        # step particles forward
        for p in self.particles:
            self.particle_step(p, ut, dt)

    def add_noise(self, sigmas):
        self.particles = np.random.normal(self.particles, sigmas)

    def particle_step(self, p, ut, dt):
       # p is a 1x3 array (x, y, theta)
       p[0] = p[0] + cos(p[2]) * ut[0] * dt
       p[1] = p[1] + sin(p[2]) * ut[0] * dt
       p[2] = p[2] + ut[2] * dt

    def particle_preview(self, p, ut, dt):
        return [p[0] + cos(p[2]) * ut[0] * dt,
                p[1] + sin(p[2]) * ut[0] * dt,
                p[2] + ut[2] * dt]

    def particle_sample(self, tmp, weights):
        r = np.random.random_sample() * sum(weights)
        for i, w in enumerate(weights):
            r -= w
            if r < 0:
                break
        return tmp[i]
        
class EKF(object):
    """The extended Kalman filter

    Attributes:
        state: the pose [x, y, theta]
        sigma: sensor sigmas for range, bearing, signature
        covariance: sigmas for state
    """
    def __init__(self, state=[], e_motion=[], e_measure=[]):
        self.state = np.array(state)
        self.e_measure = e_measure
        self.e_motion = e_motion

        # set initial covariance matrix (guess)
        self.covariance = np.array([[0., 0., 0.],
                                    [0., 0., 0.],
                                    [0., 0., 0.]])
    def update(self, ut, dt, zt):
        """Step forward with the particle filter algorithm

        See Table 7.2

        Args:
            ut: the commanded control
            dt: the time to apply the control
            zt: the measurement data as a list of features
        """
        np.array(self.state)
        theta = self.state[2]

        sigmar = self.e_measure[0]
        sigmaphi = self.e_measure[1]
        sigmas = self.e_measure[2]

        alpha1 = self.e_motion[0]
        alpha2 = self.e_motion[1]
        alpha3 = self.e_motion[2]
        alpha4 = self.e_motion[3]

        vt = ut[0]
        wt = ut[2]
        zero_command = False
        if wt == 0.0: # TODO: fix this hack
            wt = 0.1
            zero_command = True

        Gt = np.array([[1, 0, -vt/wt*np.cos(theta) + vt/wt*np.cos(theta + wt*dt)],
                       [0, 1, -vt/wt*np.sin(theta) + vt/wt*np.sin(theta + wt*dt)],
                       [0, 0, 1                                                 ]])
        Vt = np.array([[(-np.sin(theta) + np.sin(theta + wt*dt)) / wt,  vt*(np.sin(theta)-np.sin(theta+wt*dt))/pow(wt, 2) + vt*np.cos(theta+wt*dt)*dt/wt],
                       [( np.cos(theta) - np.cos(theta + wt*dt)) / wt, -vt*(np.cos(theta)-np.cos(theta+wt*dt))/pow(wt, 2) + vt*np.sin(theta+wt*dt)*dt/wt],
                       [                         0             ,                           dt            ]])
        Mt = np.array([[alpha1 * pow(vt, 2) + alpha2 * pow(wt, 2), 0],
                       [0, alpha3 * pow(vt, 2) + alpha4 * pow(wt, 2)]])

        mubar = np.add(self.state, np.array([-vt/wt*np.sin(theta) + vt/wt*np.sin(theta+wt*dt),
                                           vt/wt*np.cos(theta) - vt/wt*np.cos(theta+wt*dt),
                                           wt*dt                                   ]))
        epsilonbar = np.add(Gt.dot(self.covariance).dot(Gt.T), Vt.dot(Mt).dot(Vt.T))
        Qt = np.array([[pow(sigmar, 2), 0, 0],
                       [0, pow(sigmaphi, 2), 0],
                       [0, 0, pow(sigmas, 2)]])

        # for all observed features...
        pz = 1.
        dr = 0
        dphi = 0
        for zi in zt:
            landmark = M[c(zi[2])]
            mjx = landmark[0]
            mjy = landmark[1]
            mjs = zi[2]
            qq = pow(mjx - mubar[0], 2) + pow(mjy - mubar[1], 2)
            zihat = np.array([ np.sqrt(qq), atan2(mjy - mubar[1], mjx - mubar[0]) - mubar[2], mjs])
            
            Hi = np.array([[-(mjx-mubar[0])/np.sqrt(qq), -(mjy-mubar[1])/np.sqrt(qq),  0],
                           [ (mjy-mubar[1])/qq         , -(mjx-mubar[0])/qq         , -1],
                           [ 0                         , 0                          ,  0]])
            Si = np.array(np.add(Hi.dot(epsilonbar).dot(Hi.T), Qt))
            Ki = epsilonbar.dot(Hi.T).dot(np.linalg.pinv(Si))
            dz = np.add(zi, -zihat)
            dz[1] = dz[1] % (2*pi)
            # print "dz = {}".format(dz)
            mubar = np.add(mubar, Ki.dot(dz))
            epsilonbar = np.dot(np.add(np.identity(len(self.state)), -Ki.dot(Hi)), epsilonbar)
            pz = pz*pow(np.linalg.det(2*pi*Si), -1/2)*np.exp(-1/2*dz.T.dot(np.linalg.pinv(Si)).dot(dz))
            print "pz = {}".format(pz)
        # if zero_command is True:
        self.covariance = epsilonbar
        if zero_command is True or pz < .1:
            self.state[0] = self.state[0] + cos(self.state[2]) * ut[0] * dt
            self.state[1] = self.state[1] + sin(self.state[2]) * ut[0] * dt
            self.state[2] = self.state[2] + ut[2] * dt
            return 
        self.state = mubar
        return self.state, self.covariance

##################################################
# Measurement Model (Likelihood) Functions
##################################################

def importance_factor(fz, phat):
    """Compute the likelihood of a measurement z

    Args:
        fz: the measurement
        phat: a pose
    Returns:
        The probability p(f(z) | phat, M )
    """
    l = 1.
    for fi in fz:
        l = l * compute_feature_likelihood(fi, phat)
    return l

def compute_feature_likelihood(f, phat):
    """Computes the likelihood of a single feature f

    Args:
        f: the feature
        phat: a hypothetical pose [x,y,heading]

    Returns:
        The probability p(f | phat, M )
    """
    sigmar = 2
    sigmaphi = pi / 32
    landmark = M[c(f[2])]

    rhat = sqrt(pow(landmark[0] - phat[0], 2) + pow(landmark[1] - phat[1], 2))
    phihat = atan2(landmark[1] - phat[1], landmark[0] - phat[0]) - phat[2]

    dr = f[0] - rhat # diff in range
    dphi = acos(cos(f[1]) * cos(phihat) + sin(f[1]) * sin(phihat)) # diff in bearing
    return norm_pdf(dr, sigmar) * norm_pdf(dphi, sigmaphi)

def norm_pdf(v, sigma):
    """The probability density function of a univariate zero-mean normal distribution.

    Args:
        v: the value for which the probability shall be evaluated
        sigma: the standard devation sigma > 0.
    """
    return 1 / sqrt(2*pi*pow(sigma, 2)) * exp(-pow(v, 2) / (2*pow(sigma, 2)))
