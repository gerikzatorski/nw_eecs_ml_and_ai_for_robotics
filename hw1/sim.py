import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from graphs import GridGraph
from planners import BFS, AStar
from mwr import Unicycle

class Simulation(object):
    def __init__(self, graph=None, planner=None, robot=None, obstacles=True):
        """Init simulation based objects and set start/goal for planner and robot"""
        self.graph = graph
        self.planner = planner
        self.planner.graph = self.graph
        self.robot = robot

    def setup(self, start, goal):
        self.start = start
        self.goal = goal

        self.planner.set_start(self.start)
        self.planner.set_goal(self.goal)
        self.robot.set_goal(self.goal)

    def place_obstacles(self, extend=False):
        """Place obstacles in graph"""
        for i, o in enumerate(np.loadtxt('ds1/ds1_Landmark_Groundtruth.dat')):
            self.graph.add_obstacle((o[1], o[2]), extend=extend)
    
    def run(self):
        # run planner
        path = self.planner.plan()

        # Setup and run robot
        # self.robot.drive_basic(path)

    def display(self, dplan=True, drobot=True, real_coords=True, pretty=False, debug=False):
        fig, ax = plt.subplots()
        ax.grid()

        xdim = self.graph.celldim[0]
        ydim = self.graph.celldim[1]

        # ax.set_xlim(self.graph.xlim[0] / xdim, self.graph.xlim[1] / xdim)
        # ax.set_ylim(self.graph.ylim[1] / ydim, self.graph.ylim[0] / ydim)
        
        ## Transform Axis to represent real world coordinates instead of grid coordinates
        if real_coords:
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format( (x + xdim) * xdim + self.graph.xlim[0] - xdim / 2. + xdim / 2. ))
            ax.xaxis.set_major_formatter(ticks_x)
            ticks_y = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format( (y + ydim) * ydim + self.graph.ylim[0] - ydim / 2. ))
            ax.yaxis.set_major_formatter(ticks_y)
    
        img = np.zeros( shape=(self.graph.nrow, self.graph.ncol) )
        for x in range(self.graph.ncol):
            for y in range(self.graph.nrow):
                img[y, x] = self.graph.data[self.graph.onedim((x,y))].status
        ax.imshow(img, cmap='Greys',  interpolation='nearest')

        # draw planned path from node paths in planner
        if dplan:
            xvals = [self.graph.coord_to_2D(x.position)[0] for x in self.planner.path]
            yvals = [self.graph.coord_to_2D(x.position)[1] for x in self.planner.path]
            ax.plot(xvals, yvals, color='y')
            if not pretty: ax.scatter(xvals, yvals, color='y')

        # draw robot path
        if drobot:
            xvals = [self.graph.coord_to_2D((x[0], x[1]))[0] for x in self.robot.path]
            yvals = [self.graph.coord_to_2D((x[0], x[1]))[1] for x in self.robot.path]
            ax.plot(xvals, yvals, color='b')
            if not pretty: ax.scatter(xvals, yvals, color='b')

        # draw start and end
        start = self.graph.coord_to_2D(self.start)
        goal = self.graph.coord_to_2D(self.goal)
        ax.scatter(start[0], start[1], color='r')
        ax.scatter(goal[0], goal[1], color='g')

        # debug: 1D indices to plot
        if debug == True:
            for i in range(len(self.graph.data)):
                px, py = self.graph.twodim(i)
                ax.text(px, py, str(i))

        plt.show()
