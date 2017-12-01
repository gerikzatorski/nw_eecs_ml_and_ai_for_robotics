import numpy as np
import matplotlib.lines as lines
from Queue import Queue, PriorityQueue

from graphs import Node
from tools import dist
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

    def plan():
        """Plans a path. Algorithms implemented in subclasses

        Returns:
            A list of Nodes representing the planned path
        """
        pass
    
    def set_start(self, pos):
        """Set the planner start position with a real world coordinate"""
        if self.graph.node_at(pos):
            self.start = self.graph.node_at(pos)
        else:
            raise Exception("Planner must start at a grid cell center, not {}".format(pos))

    def set_goal(self, pos):
        """Set the planner goal position with a real world coordinate"""
        if self.graph.node_at(pos):
            self.goal = self.graph.node_at(pos)
        else:
            raise Exception("Planner must end at a grid cell center, not {}".format(pos))

class BFS(Planner):
    """Breadth First Search"""
    def plan(self):
        done = False
        q = Queue()
        q.put(self.start)
        while not q.empty() and not done:
            item = q.get()
            if item.visited:
                continue
            item.visited = True
            for n in self.graph.neighbors(item):
                if n.parent is None:
                    n.parent = item
                if n is self.goal:
                    print "GOAL REACHED!"
                    done = True
                    break
                q.put(n)
        if done:
            n = self.goal
            path = []
            while n is not self.start:
                path.append(n)
                n = n.parent
            path.append(self.start)
            self.path = path
        
class AStar(Planner):
    """A-Star, the end all be all of path planning algorithms (for now)

    The algorithms maintains two sets of notes:
        openset: List of potential nodes for best path. If emtpy, there is no path.
        closedset: List of nodes that have been considered/visited already
    """
    def plan(self):
        # for node in self.graph.data:
            # node.f = 0

        openset = [self.start]
        closedset = []

        self.start.g = 0
        self.start.f = 0
        
        while len(openset) > 0:
            # print "openset = {}".format(openset)
            # print "closedset = {}".format(closedset)

            # find node with lowest f
            # TODO: replace this with priorityqueue, fibonacci heap
            cheapest = openset[0]
            index = 0
            for i, n in enumerate(openset):
                if n.f <= cheapest.f:
                    cheapest = n
                    index = i

            item = openset.pop(index)
            # print "Testing node {}".format(item.name)
            if item is self.goal:
                print "GOAL REACHED!"
                return self.construct_path(self.goal)
            closedset.append(item)

            for neighbor in self.graph.neighbors(item):
                neighbor.g = dist(neighbor.position, item.position)
                if neighbor.parent is None and neighbor is not self.start:
                    neighbor.parent = item
                if neighbor not in closedset:
                    neighbor.f = neighbor.g + dist(neighbor.position, self.goal.position)
                    if neighbor not in openset:
                        openset.append(neighbor)
                    else:
                        openneighbor = openset[openset.index(neighbor)]
                        if neighbor.g < openneighbor.g:
                            openneighbor.g = neighbor.g
                            openneighbor.parent = neighbor.parent
        return False

    def construct_path(self, node):
        """Returns the solved path as a list [start_node, ..., goal_node]"""
        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        self.path = list(reversed(path))
        return self.path

# Unit Tests
if __name__ == "__main__":
    pass
