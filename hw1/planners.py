import numpy as np
from queue import PriorityQueue

from tools import heuristic_diagonal, heuristic_euclidean

class Planner(object):
    """General planner class

    Attributes
        graph: graph data structure to plan on
        start: The start node
        goal: The goal node
        path: A list of nodes in a successfully planned path
    """
    def __init__(self, graph=None):
        self.graph = None
        self.start = None
        self.goal = None
        self.path = None

    def plan(self):
        """Plans a path. Algorithms implemented in subclasses

        Returns:
            A list of Nodes representing the planned path
        """
        raise NotImplementedError('Subclasses need to implement their algorithm')
    
    def set_start(self, coord):
        """Set the planner start position 

        Args:
            coord: start position coordinates [x, y]
        """
        self.start = self.graph[coord]

    def set_goal(self, coord):
        """Set the planner goal position
        
        Args:
            coord: goal position coordinates [x, y]
        """
        self.goal = self.graph[coord]

class AStar(Planner):
    """A-Star, the end all be all of path planning algorithms"""
    def plan(self):
        q = PriorityQueue()
        q.put((0.0, self.start))

        parents = {}
        costs = {}

        parents[self.start] = None
        costs[self.start] = 0

        while not q.empty():
            _, curr = q.get()
            
            if curr is self.goal:
                print('Found Path!')
                return self.construct_path(self.goal, parents)
            
            for neighbor in self.graph.neighbors(curr):
                # partial observability allows robot to detect obstacles
                # within 8 neighbor cells (ie. next to planner start)
                if parents[curr] == None and neighbor.occupied:
                    neighbor.cost_observed = 1000

                g = costs[curr] + neighbor.cost_observed
                
                if neighbor not in costs or g < costs[neighbor]:
                    costs[neighbor] = g
                    f = g + heuristic_euclidean(neighbor, self.goal)
                    q.put((f, neighbor))
                    parents[neighbor] = curr
                    
        return False

    def construct_path(self, node, parents):
        """Returns the solved path as a list [start_node, ..., goal_node]

        Args:
            node: the goal node
            parents: dictionary of nodes' parents from algorithms
        """
        path = [node]
        while parents[node] is not None:
            node = parents[node]
            path.append(node)
        self.path = list(reversed(path))
        return self.path
