from math import cos, sin, atan2

class Vector(object):
    def __init__(self, x=0., y=0.):
        self.x = x
        self.y = y

    def __str__(self):
        return "Vector [{}, {}]".format(self.x, self.y)

    def __rmul__(self, other):
        return Vector(self.x*other, self.y*other)

    def __mul__(self, other):
        return Vector(self.x*other, self.y*other)
        
    def normalize(self):
        l = sqrt(self.x*self.x + self.y*self.y)
        self.x = self.x / l
        self.y = self.y / l

class Twist(object):
    def __init__(self, x=0., y=0., theta=0.):
        self.linear = Vector(x, y)
        self.angular = theta

    def __str__(self):
        return "Twist {{linear: {0!s}, angular: {1!s}}}".format(self.linear, self.angular)

class Pose(object):
    def __init__(self, x, y, orientation):
        self.position = Vector(x, y)
        self.orientation = orientation

    def __str__(self):
        return "Pose {{position: {0!s}, orientation: {1!s}}}".format(self.position, self.orientation)

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
    
# Helper Functions
def angle_to_vector(theta):
    return Vector(cos(theta), sin(theta))

def vector_to_angle(V):
    return atan2(V.y, V.x)