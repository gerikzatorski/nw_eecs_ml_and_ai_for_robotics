import numpy as np
import copy

from math import cos, sin, acos, exp, sqrt, atan2, pi
from scipy import stats

import config

from data import read_landmark_gt, read_barcodes

from tools import particle_preview, particle_sample

file_prefix = "ds{}/ds{}_".format(config.DATA_SET, config.DATA_SET)
landmarks = read_landmark_gt(file_prefix + 'Landmark_Groundtruth.dat')
subject_lookup = read_barcodes(file_prefix + 'Barcodes.dat')

def pf_general(particles, weights, cmd, dt, zt):
    """The particle filter algorithm, a variant of the Bayes filter based on importance sampling

    This is the algorithm in Probabilistic Robotics

    Args:
        particles: collection of particles in the last step
        ut: the control data at time t (in this case we use step time to run fake control step)
        zt: the measurement data as a list of features
    """
    M = len(particles)
    for i in range(M):
        xt = particle_preview(particles[i], cmd, dt)
        weights[i] = importance_factor(zt, xt)

    normalize_weights(weights)
    # if degeneracy is too high, resample # todo
    if 0.5 > 1/sum(np.square(weights)) / M:
        # tmp = copy.deepcopy(particles)
        tmp = np.copy(particles)
        # tmp = particles
        for i in range(M):
            psample = particle_sample(tmp, weights)
            particles[i] = psample
        normalize_weights(weights)
    
def importance_factor(fz, phat):
    """Compute the likelihood of a measurement z
    Calculates likelihood by iterating over measurement's features.

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
        phat: a hypothetical Pose object

    Returns:
        The probability p(f | phat, M )
    """
    sigmar = 0.2
    sigmaphi = pi / 32
    landmark = landmarks.get(subject_lookup.get(f.barcode))
    mjx = landmark.x
    mjy = landmark.y
    
    rhat = sqrt(pow(mjx - phat.position.x, 2) + pow(mjy - phat.position.y, 2))
    phihat = atan2(mjy - phat.position.y, mjx - phat.position.x) - phat.orientation
    dr = f.r - rhat # diff in range
    dphi = acos(cos(f.phi) * cos(phihat) + sin(f.phi) * sin(phihat)) # diff in bearing
    # return stats.norm.pdf(0, loc=dr, scale=sigmar) * stats.norm.pdf(0, loc=dphi, scale=sigmaphi)
    return norm_pdf(dr, sigmar) * norm_pdf(dphi, sigmaphi)

def norm_pdf(v, sigma):
    """The probability density function of a univariate zero-mean normal distribution.

    Args:
        v: the value for which the probability shall be evaluated
        sigma: the standard devation sigma > 0.
    """
    return 1 / sqrt(2*pi*pow(sigma, 2)) * exp(-pow(v, 2) / (2*pow(sigma, 2)))
    
def normalize_weights(weights):
    tot = sum(weights)
    for i in range (0,len(weights)):
        weights[i] = weights[i] / tot

def p_weights(weights):
    nw = []
    for i in range(0, len(weights)):
        nw.append(float("{0:.4f}".format(weights[i])))
    print nw
