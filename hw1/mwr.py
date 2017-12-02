from math import sqrt, cos, sin, atan2, pi

class MobileWheeledRobot(object):
    def __init__(self):
        pass
        
    def set_start(self, pos):
        pass
        
    def set_goal(self, pos):
        self.goal = pos[0], pos[1]

    def goal_reached(self, proximity=(0.1, 0.1)):
        if abs(self.q[0]-self.goal[0]) > proximity[0] and abs(self.q[1]-self.goal[1]) > proximity[1]:
            return False
        return True

class Unicycle(MobileWheeledRobot):
    """A simple unicycle based robot
    
    Attributes:
        q: state of the robot [x, y, heading] (m and rad)
        v: robot velocities, [forward, lateral, angular] (m/s and rad/s)
        maxacc: acceleration limits, [forward, lateral, angular] (m/s^2 and rad/s^2)
        dt: sample rate (seconds)

       _u: commanded Twist
    """
    def __init__(self, q=[0,0,0], dt=0.1, accel_max=[0,0,0]):
        self.q = q
        self.accel_max = accel_max
        self.dt = dt

        # velocities (vx, vy, angular vel)
        self.v = [0,0,0]
        
        self._u = Twist()
        # self._u = [0,0,0] # twist vector alternative
        
        self.path = []

    def set_start(self, pos):
        self.q[0] = pos[0]
        self.q[1] = pos[1]
    
    def drive(self, path):
        pass
        
    def set_command(self, twist):
        self._u = twist

    def control_step(self, dt):
        """Apply kinematic model through a single control step
        Limits accelerations before applying kinematic equations
        """
        acc = [min(self._u[i], self.accel_max[i]) for i in range(3)]
        self.q[0] = self.q[0] + cos(self.q[2]) * acc[0] * dt
        self.q[1] = self.q[1] + sin(self.q[2]) * acc[0] * dt
        self.q[2] = self.q[2] + acc[2] * dt
        self.path.append((self.q[0], self.q[1]))
        print self.q

    def drive_basic(self, path):
        """Incremental path following (ie. turn-move-turn)"""
        target = path[0].position
        
        step = 0
        idp = 0 # current paths[] index
        while not self.goal_reached():
            if step > 50: break
            if dist(self.q, target) < 0.1:
                idp += 1
                target = path[idp].position
                continue
            # align direction first
            err_theta = self.q[2] - atan2(path[idp+1].position[1] - target[1], path[idp+1].position[0] - target[0])
            if abs(err_theta) > pi / 36:
                print "rotating"
                print "err_theta = {}".format(err_theta)
                twist = Twist(0,0,-err_theta)
            else:
                print "moving"
                # move forward
                twist = Twist(0.1,0,0)
            # step increment
            step += 1
            self.set_command(twist)
            self.control_step(0.1)
        
    # def drive_path(self, path):
    #     goal = path[len(path)-1]

    #     closest = path[0]
    #     ip = 0
    #     l = dist(self.q, path[0])

    #     step = 0;
    #     while abs(self.q[0]-goal[0]) > 0.1 and abs(self.q[1]-goal[1]) > 0.1:
    #         # find nearest point (brute force)
    #         for i, p in enumerate(path):
    #             if dist(self.q, p) <= dist:
    #                 ip = i
    #                 closest = p
    #                 l = dist(self.q, p)

    #         # if last point in path
    #         if ip == len(path):
    #             pass
    #         else:
    #             cs = 0
    #             dtheta = self.q[2] - atan2(path[ip+1][1] - path[ip][1], path[ip+1][0] - path[ip][0])
            
    #             sdot = self.v[0] * cos(state[2]) / (1 - cs * l)
    #             ldot = self.v[0] * sin(state[2])
    #             dthetadot = self.q[2] - (self.v[0] * cos(dtheta) * cs) / (1 - cs * l)

    #             velocityd = 
    #             omegad = 
                
    #         self.set_command(cmd)
    #         self.control_step()
    #         step += 1


class Twist(object):
    def __init__(self, x=0., y=0., theta=0.):
        self.linear = Vector(x, y)
        self.angular = theta

    def __getitem__(self, i):
        if i == 0: return self.linear.x
        if i == 1: return self.linear.y
        if i == 2: return self.angular
                   
    def __str__(self):
        return "Twist {{linear: {0!s}, angular: {1!s}}}".format(self.linear, self.angular)

def dist(p1, p2):
    return sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))
