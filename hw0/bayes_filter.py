import numpy as np

from math import cos, sin, acos, exp, sqrt, atan2, pi

# Reading Data using config files
import config
from data import read_landmark_gt, read_barcodes
file_prefix = "ds{}/ds{}_".format(config.DATA_SET, config.DATA_SET)
landmarks = read_landmark_gt(file_prefix + 'Landmark_Groundtruth.dat')
subject_lookup = read_barcodes(file_prefix + 'Barcodes.dat')

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
        """Update the particle filter algorithm

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
        
def importance_factor(fz, phat):
    """Compute the likelihood of a measurement z

    Args:
        z: the measurement
        phat: a pose
        M: a map
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
    landmark = landmarks.get(subject_lookup.get(f.barcode))
    
    rhat = sqrt(pow(landmark[0] - phat[0], 2) + pow(landmark[1] - phat[1], 2))
    phihat = atan2(landmark[1] - phat[1], landmark[0] - phat[0]) - phat[2]

    dr = f.r - rhat # diff in range
    dphi = acos(cos(f.phi) * cos(phihat) + sin(f.phi) * sin(phihat)) # diff in bearing
    return norm_pdf(dr, sigmar) * norm_pdf(dphi, sigmaphi)

def norm_pdf(v, sigma):
    """The probability density function of a univariate zero-mean normal distribution.

    Args:
        v: the value for which the probability shall be evaluated
        sigma: the standard devation sigma > 0.
    """
    return 1 / sqrt(2*pi*pow(sigma, 2)) * exp(-pow(v, 2) / (2*pow(sigma, 2)))
    
# def normalize_weights(weights):
#     tot = sum(weights)
#     for i in range (0,len(weights)):
#         weights[i] = weights[i] / tot
