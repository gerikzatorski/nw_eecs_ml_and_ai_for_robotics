import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

from pathlib import Path

def heuristic_euclidean(current, goal):
    return np.linalg.norm(current.coord - goal.coord)

def heuristic_diagonal(current, goal):
    dx = abs(current.coord[0] - goal.coord[0])
    dy = abs(current.coord[1] - goal.coord[1])
    return max(dx, dy)

def wrap_to_pi(theta):
    """Wrap angles to range [-pi, pi] radians"""
    theta = theta % (2 * np.pi)
    if theta > np.pi:
        theta -= 2 * np.pi
    return theta

class Animator(object):
    """A tool to create and save animations out of captured plt images

    Attributes:
        img_dir: the directory path where animations are saved
        frames_dir: a subdirectory to temporarily store captured images
        out_name: output filename of animation
        dt: time difference between each figure
        interval: rate at which to capture images (None indicates inactive)
        count: tracks count of all figures processed
        flist: a list of captured images (to be cleaned)
    """
    def __init__(self, img_dir, out_name, dt=0.1, interval=None):
        # setup paths
        self.img_dir = Path(img_dir)
        self.frames_dir = self.img_dir  / 'frames'
        self.out_name = out_name + '.gif'

        self.count = 0
        self.interval = interval
        self.dt = dt
        self.flist = []

        # create frames directory
        Path(self.frames_dir).mkdir(parents=True, exist_ok=True)
        
    def capture(self):
        """Capture a snapshot of current plt figure"""
        if not self.interval:
            pass
        else:
            fname = self.frames_dir / f'{self.count}.png'
    
            # save frames at specified interval
            if self.count % self.interval == 0:
                self.flist.append(fname)
                plt.savefig(fname)
    
            # increment for all processed figures
            self.count += 1

    def save_animation(self):
        """Build gif from captured images and save"""
        if not self.interval:
            pass
        else:
            fps = self.dt / self.interval * 100
            with imageio.get_writer(self.img_dir / self.out_name, mode='I', fps=fps) as writer:
                for fname in self.flist:
                    image = imageio.imread(fname)
                    writer.append_data(image)
        
    def clean(self):
        """Delete snapshot images"""
        for fname in set(self.flist):
            os.remove(fname)
