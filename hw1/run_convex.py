from sim import Simulation
from graphs import GridGraph
from planners import AStar
from mwr import Unicycle

if __name__ == '__main__':
    
    sim = Simulation(name='convex',
                     graph=GridGraph(xlim=(-2, 5), ylim=(-6, 6), celldim=(1,1)),
                     planner=AStar(),
                     robot=Unicycle(accel_max=[0.288, 5.579]))

    sim.setup((1.5, 4.5), (1.5, -4.5))
        
    sim.place_obstacles(extend=None, landmark_file='dsx/convex_Landmark_Groundtruth.dat')

    sim.run(animation_interval=5)
