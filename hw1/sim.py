import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from tools import Animator

class Simulation(object):
    """Object used to run simulations with planning algorithms on graph structures

    Atrributes:
        name: simulation name used to save files
        graph: node based graph structure
        planner: planning algorithm
        robot: kinematic controller
        dt: time step size
        fig: matplotlib figure used for visualization
        ax: matplotlib axes used for visualization
    """
    def __init__(self, name='sim', graph=None, planner=None, robot=None, dt=0.1):
        """Init simulation based objects and set start/goal for planner and robot"""
        self.name = name
        self.graph = graph
        self.planner = planner
        if self.planner:
            self.planner.graph = self.graph
        self.robot = robot
        self.dt = dt
        
        self.fig = None
        self.ax = None

    def setup(self, start, goal):
        """Setup start and goal locations for sim

        Args:
            start: start coordinates [x, y]
            goal: goal coordinates [x, y]
        """
        # align start and goal with center of node covering coords
        self.start = self.graph[start].coord
        self.goal = self.graph[goal].coord

        if self.robot:
            self.robot.place(np.append(self.start, [-np.pi/2]))

    def place_obstacles(self, extend=None, landmark_file='ds1/ds1_Landmark_Groundtruth.dat'):
        """Place obstacles in graph
        
        Args:
            extend: how far to extend obstacle (m)
            landmark_file: landmarks filepath
        """
        for i, o in enumerate(np.loadtxt(landmark_file)):
            self.graph.add_obstacle((o[1], o[2]), extend=extend)
            
    def run(self, animation_interval=None):
        """Run the simulation

        Args:
            animation_interval: image capture rate
        """
        if not self.planner or not self.robot:
            raise Exception('Cannot simulate without planner or motion model')

        print('Running Simulation', self.name)
        
        self.init_display()

        # inactive if interval is None
        animator = Animator('img', out_name=f'animation_{self.name}',
                            dt=self.dt, interval=animation_interval)

        # outer loop terminates when robot reaches goal
        while not self.close_to(self.goal):

            # online planner replans once it gets close to each target
            self.planner.set_start(self.robot.pose[:2])
            self.planner.set_goal(self.goal)

            # extract and target first waypoint of plan
            path = self.planner.plan()
            target = np.array(path[1].coord)

            # illustrate new plan and observed obstacles
            self.draw_plan(segment=True)
            self.draw_obstacles()

            # inner loop terminates when robot reaches next target
            while not self.close_to(target):
                self.draw_robot()

                animator.capture()
                
                self.robot.update_control(target)
                self.robot.control_step(self.dt)

        animator.save_animation()
        animator.clean()
        plt.savefig(f'img/img_{self.name}.png')
        plt.show()

    def close_to(self, coord):
        """Determine if robot is within 1/20 of celldim to given location

        Args:
            coord: coordinates to check against robot location [x, y]
        """
        return (abs(self.robot.pose[0] - coord[0]) < self.graph.celldim[0]/20 and
                abs(self.robot.pose[1] - coord[1]) < self.graph.celldim[1]/20)
        
    def init_display(self):
        self.fig, self.ax = plt.subplots()

        # local copy for easier reading
        xlim = self.graph.xlim
        ylim = self.graph.ylim
        celldim = self.graph.celldim

        # adjust figure size and aspect ratio
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal')
        
        # draw minor grid ticks - aligned with cells
        self.ax.set_xticks(np.arange(xlim[0], xlim[1]+celldim[0], celldim[0]), minor=True)
        self.ax.set_yticks(np.arange(ylim[0], ylim[1]+celldim[1], celldim[1]), minor=True)
        self.ax.grid(alpha=0.3, which='minor')

        # draw major grid ticks - all integers
        self.ax.set_xticks(np.arange(xlim[0], xlim[1]+celldim[0], 1), minor=False)
        self.ax.set_yticks(np.arange(ylim[0], ylim[1]+celldim[1], 1), minor=False)
        self.ax.grid(alpha=0.7, which='major')

        # draw unobserved obstacles
        for n in self.graph.nodes:
            if n.occupied:
                self.ax.add_patch(Rectangle(xy=n.coord - celldim/2,
                                            width=celldim[0],
                                            height=celldim[1],
                                            facecolor='black'))

        # draw start and goal
        if self.planner:
            self.ax.plot(self.start[0], self.start[1],
                         marker='o',
                         color='blue')
            self.ax.plot(self.goal[0], self.goal[1],
                         marker='o',
                         color='green')

    def draw_obstacles(self):
        """ Draw obstacles as rectangles over cells"""
        for n in self.graph.nodes:
            if n.cost_observed > 1:
                self.ax.add_patch(Rectangle(xy=n.coord - self.graph.celldim/2,
                                            width=self.graph.celldim[0],
                                            height=self.graph.celldim[1],
                                            facecolor='red'))

        
    def draw_plan(self, segment):
        """Draw the entire path or first segment of the plan"""
        if self.planner:
            if segment:
                xvals = [n.coord[0] for n in self.planner.path[:2]]
                yvals = [n.coord[1] for n in self.planner.path[:2]]
            else:
                xvals = [n.coord[0] for n in self.planner.path]
                yvals = [n.coord[1] for n in self.planner.path]
                
            self.ax.plot(xvals, yvals, color='y')

    def draw_robot(self):
        """Draw a triangle marker to represent the robot pose"""
        if self.robot:
            self.ax.plot(self.robot.pose[0], self.robot.pose[1],
                         marker=(3, 0, np.degrees(self.robot.pose[2] - np.pi/2)),
                         markersize=3,
                         color='black',
                         # markeredgecolor='black',
                         # markeredgewidth=0.3,
                         linestyle ='None')
