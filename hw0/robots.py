from math import cos, sin, pi
from tools import Vector, Pose, Twist, angle_to_vector
from copy import deepcopy

import numpy as np

class Robot(object):
    
    def __init__(self, position=Vector(0,0), orientation=0, noisy=False, color='k'):
        
        # pose
        self.position = deepcopy(position)
        self.orientation = orientation

        # velocities
        self.velocity = Vector(0,0)
        self.angular_velocity = 0

        self._command = Twist()

        self.color = color

        if noisy:
            self.add_noise()

    def __str__(self):
        return "Robot [{}, {}]".format(self.position, self.orientation)

    # def __deepcopy__(self, memo):
    #     dpcpy = Robot(position=self.position, orientation=self.orientation, color=self.color)
    #     dpcpy._command = copy.deepcopy(self._command)
    #     dpcpy.velocity = copy.deepcopy(self.velocity)
    #     dpcpy.angular_velocity = self.angular_velocity
    #     return dpcpy
                
    def control_step(self, dt):
        self.velocity = angle_to_vector(self.orientation) * self._command.linear.x
        self.angular_velocity = self._command.angular

        # kinematic model (todo: move dead-reckoning outside of robot model)
        self.position.x = self.position.x + (self.velocity.x * dt)
        self.position.y = self.position.y + (self.velocity.y * dt)
        self.orientation = self.orientation + (self.angular_velocity * dt)

    def next_pose(self, dt):
        speed = self._command.linear.x
        velocity = angle_to_vector(self.orientation) * speed
        angular_velocity = self._command.angular

        # kinematic model (todo: move dead-reckoning outside of robot model)
        x = self.position.x + (velocity.x * dt)
        y = self.position.y + (velocity.y * dt)
        orientation = self.orientation + (angular_velocity * dt)
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
