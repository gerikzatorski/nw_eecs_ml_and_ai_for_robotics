import numpy as np

class Unicycle(object):
    """A simple unicycle based robot
    
    Attributes:
        state: state of the robot [x, y, heading] (m, m, rad)
       _u: commanded velocities [forward, lateral, angular] (m/s, m/s, rad/s)
    """
    def __init__(self, state=[0,0,0]):
        self.state = np.array(state)
        self._u = [0,0,0]

    def __str__(self):
        return f"Robot [{self.position}, {self.orientation}]"

    def control_step(self, dt):
        """Apply kinematic model through a single control step
        """
        self.state[0] += np.cos(self.state[2]) * self._u[0] * dt
        self.state[1] += np.sin(self.state[2]) * self._u[0] * dt
        self.state[2] += self._u[2] * dt

    def set_command(self, ut):
        self._u = ut

