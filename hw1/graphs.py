import numpy as np

class Graph(object):
    """Barebones graph abstraction
        nodes: 1D array of nodes
        edges: 1D array of edges
    """
    def __init__(self, nodes=[], edges=[]):
        self.nodes = nodes
        self.edges = edges

    def neighbors(self, node):
        """Return list of neighbors in node form"""
        raise NotImplementedError('Subclasses need to implement this function')

class GridGraph(Graph):
    """Graph abstraction for a 2D grid

    Variables typically used to index each type of coordinate system...
    real world coordinates, coord = (x, y)
    2D grid, indices = (idx, idy)
    1D list, i

    Atrributes:
        xlim: the grid's x boundaries [min, max]
        ylim: the grid's y boundaries [min, max]
        nrow: number of rows in grid
        ncol: number of columns in grid
        celldim: real world dimensions of each grid cell [width, height]
        nodes: the list of Node objects
    """
    def __init__(self, xlim=(0,3), ylim=(0,2), celldim=(1,1)):
        self.xlim = np.array(xlim)
        self.ylim = np.array(ylim)
        self.celldim = np.array(celldim)
        
        self.nrow = int((ylim[1] - ylim[0])/celldim[1])
        self.ncol = int((xlim[1] - xlim[0])/celldim[0])

        # create nodes
        xmin = xlim[0] + celldim[0]/2
        ymin = ylim[0] + celldim[1]/2
        self.nodes = []
        for y in range(self.nrow):
            for x in range(self.ncol):
                n = Node(coord=(xmin + x * celldim[0], ymin + y * celldim[1]))
                self.nodes.append(n)
        
        # create edges (ie. neighbors)
        self.edges = []
        num_nodes = len(self.nodes)
        for i in range(num_nodes):
            neighbors = []
            idx, idy = self.twodim(i)
            for x in [idx, idx-1, idx+1]:
                for y in [idy, idy-1, idy+1]:
                    # x and y are 2D indices (not coords here)
                    if x != idx or y != idy: # ignore self
                        if self.in_grid((x, y)):
                            neighbors.append(self.onedim((x,y)))
            self.edges.append(neighbors)

    def __getitem__(self, coord):
        """Returns the Node that covers the given coordinate"""
        return self.nodes[self.coord_to_1d(coord)]

    def neighbors(self, node):
        neighbors = []
        for i in self.edges[self.coord_to_1d(node.coord)]:
            neighbors.append(self.nodes[i])
        return neighbors
    
    def coord_to_2D(self, coord):
        """Converts real world coordinates to grid indices"""
        idx = int((coord[0] - self.xlim[0]) // self.celldim[0])
        idy = int((coord[1] - self.ylim[0]) // self.celldim[1])
        return idx, idy
        
    def coord_to_1d(self, coord):
        """Converts real world coordinateso to 1D index"""
        return self.onedim(self.coord_to_2D(coord))

    def onedim(self, indices):
        """Maps 2D grid indices to 1D index"""
        return int(indices[0] + indices[1] * self.ncol)
        
    def twodim(self, i):
        """Maps 1D index to 2D grid indices"""
        idx = i % self.ncol
        idy = i // self.ncol
        return idx, idy

    def in_grid(self, indices):
        """Checks if 2D indices are within grid boundaries"""
        idx, idy = indices
        if 0 > idy or idy >= self.nrow: return False
        if 0 > idx or idx >= self.ncol: return False
        return True

    def add_obstacle(self, coord, extend=None):
        """Adds an obstacle to the the grid based on real world coordinates

        Args:
            coord: real world coordinates of obstacle
            extend: how far to extend obstacle (m)
        """
        self[coord].occupied = True

        # extend obstacle to adjacent cells (results in a square)
        if extend:
            for dx in np.arange(-extend, extend + self.celldim[0], self.celldim[0]):
                for dy in np.arange(-extend, extend + self.celldim[1], self.celldim[1]):
                    self[coord + np.array((dx,dy))].occupied = True
    
class Node(object):
    """Node structure used in graph theory

    Atrributes:
        coord: the node center
        occupied: boolean value indicating if the node is an obstacle
        cost_observed: the observed cost of traveling to this node
    """
    def __init__(self, coord=(0,0), cost_observed=1):
        self.coord = np.array(coord)
        self.occupied = False
        self.cost_observed = cost_observed

    def __str__(self):
        return f'[{str(self.coord)}={self.occupied}]'

    # comparison operators needed for PriorityQueue fallback
    def __gt__(self, other):
        return (self.coord > other.coord).any()

    def __lt__(self, other):
        return (self.coord < other.coord).any()
