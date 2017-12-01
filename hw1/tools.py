from math import sqrt, cos, sin, atan2, pi

def dist(p1, p2):
    return sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))
    
def angle_to_vector(theta):
    return Vector(cos(theta), sin(theta))

def vector_to_angle(V):
    return atan2(V.y, V.x)
