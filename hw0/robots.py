from math import cos, sin, pi
from tools import Vector, Pose, Twist, angle_to_vector

import numpy as np

class Robot(object):
    def __init__(self, position=Vector(0,0), orientation=0, noisy=False, color='k'):
        self.position = position
        self.velocity = Vector(0,0)
        self.orientation = orientation
        self.angular_velocity = 0
        self.color = color
        self._command = Twist()

        if noisy:
            self.add_noise()

        # private
    def __str__(self):
        return "Robot [{}, {}]".format(self.position, self.orientation)
        
    def control_step(self, dt):
        speed = self._command.linear.x
        self.velocity = angle_to_vector(self.orientation) * speed
        self.angular_velocity = self._command.angular

        # kinematic model (todo: move dead-reckoning outside of robot model)
        dx = self.velocity.x * dt
        dy = self.velocity.y * dt
        dtheta = self.angular_velocity * dt
        self.position.x = self.position.x + dx
        self.position.y = self.position.y + dy
        self.orientation = self.orientation + dtheta

    def next_pose(self, dt):
        speed = self._command.linear.x
        velocity = angle_to_vector(self.orientation) * speed
        angular_velocity = self._command.angular

        # kinematic model (todo: move dead-reckoning outside of robot model)
        dx = velocity.x * dt
        dy = velocity.y * dt
        dtheta = angular_velocity * dt
        x = self.position.x + dx
        y = self.position.y + dy
        orientation = self.orientation + dtheta
        return Pose(x=x, y=y, orientation=orientation)

    def add_noise(self, xSigma=0.03, ySigma=0.03, thetaSigma=pi / 32):
        self.position.x = np.random.normal(self.position.x, xSigma)
        self.position.y = np.random.normal(self.position.y, ySigma)
        self.orientation = np.random.normal(self.orientation, thetaSigma)
        
    def set_command(self, twist):
        if twist is None:
            raise ValueError("Command may not be null.")
        if twist.linear.y != 0.:
            raise ValueError("Cannot move sideways.")
        self._command = twist

    def set_position(self, position):
        self.position = position

    def set_orientation(self, orientation):
        self.orientation = orientation
        
    def get_pose(self):
        return Pose(x=self.position.x, y=self.position.y, orientation=self.orientation)
