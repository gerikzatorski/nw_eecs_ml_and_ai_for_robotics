import sys

from math import pi

from graphs import GridGraph
from planners import BFS, AStar
from mwr import Unicycle

from sim import Simulation

## MAIN ##
if __name__ == "__main__":
    
    # parse arguments
    try:
        test = int(sys.argv[1])
    except:
        test = 3

    if test in range(1,4):
        cells = (1, 1)
        extend = False
    elif test in range(4,7):
        cells = (.1, .1)
        extend = True
    else:
        print "Test number not valid"
        
    sim = Simulation(graph=GridGraph(xlim=(-2, 5), ylim=(-6, 6), celldim=cells),
                     planner=AStar(),
                     robot=Unicycle(q=[0,0,-pi/2], dt=0.1, accel_max=[0.288, 0, 5.579]))

    # for grid cells of size 1, 1
    if test == 1:
        sim.setup((0.5, -1.5), (0.5, 1.5))
    if test == 2:
        sim.setup((4.5, 4.5), (4.5, -1.5))
    if test == 3:
        sim.setup((-0.5, 5.5), (1.5, -3.5))
    # for grid cells of size 0.1, 0.1
    if test == 4:
        sim.setup((2.45, -3.55), (0.95, -1.55))
    if test == 5:
        sim.setup((4.95, -0.05), (2.45, 0.25))
    if test == 6:
        sim.setup((-0.55, 1.45), (1.95, 3.95))

    sim.place_obstacles(extend=extend)
    sim.run()
    sim.display(drobot=False, real_coords=True, pretty=True)
