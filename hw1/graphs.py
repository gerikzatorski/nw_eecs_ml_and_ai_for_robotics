import sys
import numpy as np

from math import floor

from mwr import Unicycle

class Graph(object):
    """Barebones graph abstraction
        nodes: 1D array of nodes
        edges: 1D array of edges
    """
    def __init__(self, nodes=[], edges=[]):
        self.nodes = nodes
        self.edges = edges

    def neighbors(self, x):
        return self.edges[self.nodes.index(x)]

class GridGraph(Graph):
    """
    2D grid graph represented in 1D python list
    
    Variables typically used to index each coordinate system...
    real world coordinates, coord = (x, y)
    2D grid, indices = (col, row) # todo: make it (idx, idy)
    1D list, i

    TODO: actually use the "interface" Graph class (ie. how to iterate through nodes regardless
    of what subclass is used) which will help planning algorithm execution
    

    Atrributes:
        xlim: A tuple for the grid's x boundaries (min, max)
        ylim: A tuple for the grid's y boundaries (min, max)
        celldim: real world dimensions of each grid cell (x, y)
        data: 1D list of Node objects

    """
    def __init__(self, xlim=(0,3), ylim=(0,2), celldim=(1,1)):

        self.xlim = xlim
        self.ylim = ylim
        self.celldim = celldim
        
        self.nrow = int((ylim[1] - ylim[0])/celldim[1])
        self.ncol = int((xlim[1] - xlim[0])/celldim[0])

        xmin = xlim[0] + celldim[0]/2.
        ymin = ylim[0] + celldim[1]/2.

        self.data = [0 for i in range(self.nrow*self.ncol)]
        for x in range(self.ncol):
            for y in range(self.nrow):
                i = self.onedim((x, y))
                n = Node(name=str(i), position=(xmin + x * celldim[0], ymin + y * celldim[1]))
                self.data[i] = n

    def __getitem__(self, coord):
        """Returns the Node that covers the given coordinate"""
        x, y = coord
        i = self.coord_to_1d(coord)
        # col = int(floor((x - self.xlim[0]) / self.celldim[0]))
        # row = int(floor((y - self.ylim[0]) / self.celldim[1]))
        return self.data[i]

    def __str__(self):
        # return str(np.array(self.data))
        s = ""
        for y in range(self.nrow):
            if y != 0: s += '\n'
            for x in range(self.ncol):
                s += "{: >20}".format(self.data[self.onedim((x, y))])
                # s += "{}".format(self.data[self.onedim((x, y))])
        return s
    
    def coord_to_2D(self, coord):
        """Converts real world coordinates to grid indices"""
        x, y = coord
        idx = int(floor((x - self.xlim[0]) / self.celldim[0]))
        idy = int(floor((y - self.ylim[0]) / self.celldim[1]))
        return idx, idy
        
    def coord_to_1d(self, coord):
        """Converts real world coordinateso to 1D index"""
        x, y = coord
        idx= int(floor((x - self.xlim[0]) / self.celldim[0]))
        idy = int(floor((y - self.ylim[0]) / self.celldim[1]))
        return self.onedim((idx, idy))

    def onedim(self, indices):
        """Maps 2D grid indices [idx, idy] to list index (i)"""
        # todo: what if pos is not a node.position, should i just return closest? or diff fxn for that?
        col, row = indices
        return int(col + row*self.ncol)
    
        
    def twodim(self, i):
        """Maps list index to 2D grid indices"""
        x = i % self.ncol
        y = i / self.ncol
        return x, y

    def neighbors(self, node):
        idx, idy = self.twodim(self.data.index(node))
        result = []
        for x in range(idx-1, idx+2):
            for y in range(idy-1, idy+2):
                if x != idx or y!= idy:
                    if self.in_grid((x, y)):
                        node = self.data[self.onedim((x,y))]
                        if node.status is False:
                            result.append(node)
        return result
    
    def in_grid(self, indices):
        """Checks if 2D indices are within grid boundaries"""
        idx, idy= indices
        if 0 > idy or idy >= self.nrow: return False
        if 0 > idx or idx >= self.ncol: return False
        return True

    def node_at(self, coord):
        """Checks if there is a node at a given coordinate of floats

        TODO: figure out how to do this without strings
        """
        x, y = coord
        x = str(x)
        y = str(y)
        xdim = self.celldim[0]
        ydim = self.celldim[1]

        # xvals = np.arange(self.xlim[0] + xdim/2., self.xlim[1], xdim)
        # yvals = np.arange(self.ylim[0] + ydim/2., self.ylim[1], ydim)

        xvals = [str(i) for i in np.arange(self.xlim[0] + xdim/2., self.xlim[1], xdim)]
        yvals = [str(i) for i in np.arange(self.ylim[0] + ydim/2., self.ylim[1], ydim)]

        if (x in xvals) and (y in yvals):
            return self[coord]
        return False

    def add_obstacle(self, coord, extend=True):
        """Adds an obstacle to the the grid based on real world coordinates"""
        # todo: generalize size of obstacles
        if extend:
            size = 3
            col, row = self.coord_to_2D(coord)
            for c in range(col - size, col + size):
                for r in range(row - size, row + size):
                    self.data[self.onedim((c, r))].status = True
        else:
            indices = self.coord_to_2D(coord)
            self.data[self.onedim(indices)].status = True

class Node(object):
    """Node structure used in graph theory

    Node dimensions are handled by GridGraph

    position: A 2D coordinate tuple for the node center
    status: boolean value where True indicates an obstacle in grid
    parent: Points to previous node in planner algorithm path
    visited: Indicates this node has been visited by the planner
    """
    def __init__(self, name='', position=(0,0)):
        self.name=name
        self.position = position
        self.status = False
        self.parent = None
        self.visited = False
        self.f = 0
        self.g = 0

    def __str__(self):
        s = None
        if self.parent is not None:
            s = self.parent.name
        return "[{}={}, {}]".format(self.name, self.status, s)
    
    def __repr__(self):
        return self.name

# Unit Testing
if __name__ == "__main__":

    # print "Testing GridGraph init"
    # G0 = GridGraph(xlim=(-1,2), ylim=(-1,1), celldim=(1, 1))
    G1 = GridGraph(xlim=(-1,2), ylim=(-1,1), celldim=(0.5,0.5))
    # G2 = GridGraph(xlim=(-1,2), ylim=(-1,1), celldim=(0.1, 0.1))

    # print G0
    print "++++++"
    print G1
    # print G2
    
    # G = GridGraph(xlim=(-1,2), ylim=(-1,1), celldim=(0.5,0.5))
    # G = GridGraph(celldim=(0.5,0.5))
    # print "-------"
    # print G
    # print "-------"
    # pts = [(0.3, -0.1),
    #        (-0.7, -0.8),
    #        (1.7, 0.6)]
    # for p in pts:
    #     print "Testing {}".format(p)
    #     # print "{} onedim = {}".format(p, G.onedim(p))
    #     print "{} getitem = {}".format(p, G[p])
    #     print "+++++++++++++"
    # sys.exit()
    # print G.node_at((1,1))
    # print G.node_at((0.5,0.5))
    # print G.node_at((1.5,1.5))

    # G = GridGraph(xlim=(-2, 5), ylim=(-6, 6), celldim=(.5,.5))
    # print G.onedim((1.88, -5.57))
    # print G[1.88, -5.57]
