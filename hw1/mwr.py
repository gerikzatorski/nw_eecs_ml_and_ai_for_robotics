import numpy as np

from tools import wrap_to_pi

class Unicycle(object):
    """A simple unicycle based robot
    
    Attributes:
        pose: pose of the robot [x, y, heading] (m and rad)
        accel_max: acceleration limits [forward (m/s^2), angular (rad/s^2)]

        _u: command velocities [linear (m/s), angular (rad/s)]
        _v: robot velocities [linear (m/s), angular (rad/s)]
    """
    def __init__(self, pose=[0,0,0], accel_max=[0,0]):
        self.pose = np.array(pose)
        self.accel_max = accel_max

        self._u = [0,0]
        self._v = [0,0]

    def place(self, pose):
        """Move the robot

        Args:
            pose: where to place to robot
        """
        self.pose = np.array(pose)

    def update_control(self, goal, gains=[0.4, 1.8]):
        """Calculate control and store in member attribute

        Args:
            gains: control constants [linear, angular]
        """
        dx, dy = np.subtract(goal, self.pose[:2])

        dtheta = np.arctan2(dy, dx) - self.pose[2]
        dtheta = wrap_to_pi(dtheta)
        
        self._u[0] = gains[0] * np.sqrt(pow(dx,2) + pow(dy,2))
        self._u[1] = gains[1] * dtheta

    def control_step(self, dt):
        """Apply kinematic model through a single control step

        Args:
            dt: time span of control step (s)
        """
        # calculate potential accelerations
        a = np.subtract(self._u, self._v) / dt

        # update velocities (with constrained accelerations)
        if abs(a[0]) < self.accel_max[0]:
            self._v[0] += a[0] * dt
        else:
            self._v[0] += (1 if a[0] > 0 else -1) * self.accel_max[0] * dt

        if abs(a[1]) < self.accel_max[1]:
            self._v[1] += a[1] * dt
        else:
            self._v[1] += (1 if a[1] > 0 else -1) * self.accel_max[1] * dt

        # update pose
        self.pose[0] += np.cos(self.pose[2]) * self._v[0] * dt
        self.pose[1] += np.sin(self.pose[2]) * self._v[0] * dt
        self.pose[2] += self._v[1] * dt

        # keep pose between -pi and pi
        self.pose[2] = wrap_to_pi(self.pose[2])
