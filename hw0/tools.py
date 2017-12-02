import numpy as np
from math import cos, sin, atan2

class Feature(object):
    def __init__(self, time=None, barcode=None, r=0, phi=0):
        self.time = time
        self.barcode = barcode  # subject
        self.r = r              # range
        self.phi = phi          # bearing

    def __str__(self):
        return "Feature ({}, barcode={}, x={}, y={})".format(self.time, self.barcode, self.r, self.phi)

class Landmark(object):
    def __init__(self, subject=0, x=0, y=0, t=0):
        self.subject = subject
        self.x = x
        self.y = y

    def __str__(self):
        return "Landmark (subject={}, x={}, y={})".format(self.subject, self.x, self.y)
    
def particle_step(p, twist, dt):
    # p is a 1x3 array (x, y, theta)
    p[0] = p[0] + cos(p[2]) * twist[0] * dt
    p[1] = p[1] + sin(p[2]) * twist[0] * dt
    p[2] = p[2] + twist[2] * dt

def particle_noise(p, sigmas):
    # p = (x, y, theta)
    # sigmas is also 1x3
    p[0] = np.random.normal(p[0], sigmas[0])
    p[1] = np.random.normal(p[1], sigmas[1])
    p[2] = np.random.normal(p[2], sigmas[2])

def particle_preview(p, twist, dt):
    
    return [p[0] + cos(p[2]) * twist[0] * dt,
                p[1] + sin(p[2]) * twist[0] * dt,
                p[2] + twist[2] * dt]

def particle_sample(particles, weights):
    r = np.random.random_sample() * sum(weights)
    for i, w in enumerate(weights):
        r -= w
        if r < 0:
            break
    return particles[i]
        
