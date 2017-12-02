from math import cos, sin, pi
from copy import deepcopy

import numpy as np

class Unicycle(object):
    """A simple unicycle based robot
    
    Attributes:
        q: state of the robot [x, y, heading] (m and rad)
        v: robot velocities, [forward, lateral, angular] (m/s and rad/s)
        dt: sample rate (seconds)

       _u: commanded twist
    """
    def __init__(self, q=[0,0,0], savehist=False):
        self.q = q
        self.savehist = savehist

        # velocities (vx, vy, angular vel)
        self.v = [0,0,0]
        
        self._u = [0,0,0] # twist vector alternative
        
        self.path = []

    def __str__(self):
        return "Robot [{}, {}]".format(self.position, self.orientation)

    def control_step(self, dt):
        """Apply kinematic model through a single control step
        """
        self.q[0] = self.q[0] + cos(self.q[2]) * self._u[0] * dt
        self.q[1] = self.q[1] + sin(self.q[2]) * self._u[0] * dt
        self.q[2] = self.q[2] + self._u[2] * dt
        if self.savehist:
            self.path.append(self.q)

    def set_command(self, twist):
        self._u = twist

    def get_pose(self):
        return np.array(self.q)
