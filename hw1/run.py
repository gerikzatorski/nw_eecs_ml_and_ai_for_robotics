import argparse

from sim import Simulation
from graphs import GridGraph
from planners import AStar
from mwr import Unicycle

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Localization Simulation')
    parser.add_argument('--sim_num',
                        help='run individual simulation',
                        type=int,
                        choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument('--animation',
                        help='interval between captures for animation',
                        type=int)
    args = parser.parse_args()

    if args.sim_num is not None:
        sim_list = [args.sim_num]
    else:
        sim_list = [0,1,2,3,4,5]

    for sim_num in sim_list:
        if sim_num in range(0, 3):
            cells = (1, 1)
            extend = None
        elif sim_num in range(3, 6):
            cells = (0.1, 0.1)
            extend = 0.3
    
        sim = Simulation(name=str(sim_num),
                         graph=GridGraph(xlim=(-2, 5), ylim=(-6, 6), celldim=cells),
                         planner=AStar(),
                         robot=Unicycle(accel_max=[0.288, 5.579]))
    
        # for grid cells of size 1, 1
        if sim_num == 0:
            sim.setup((0.5, -1.5), (0.5, 1.5))
        if sim_num == 1:
            sim.setup((4.5, 4.5), (4.5, -1.5))
        if sim_num == 2:
            sim.setup((-0.5, 5.5), (1.5, -3.5))
    
        # for grid cells of size 0.1, 0.1
        if sim_num == 3:
            sim.setup((0.5, -1.5), (0.5, 1.5))
        if sim_num == 4:
            sim.setup((4.5, 4.5), (4.5, -1.5))
        if sim_num == 5:
            sim.setup((-0.5, 5.5), (1.5, -3.5))

        sim.place_obstacles(extend=extend)
            
        sim.run(animation_interval=args.animation)
