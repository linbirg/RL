"""
迷宫的模拟环境
"""
import random
import numpy as np
# from logger import Logger

from env.maze import Maze
from env.maze_viewer import MazeViewer

# from process_maze import ProcessMaze


class MazeEnv(object):
    """
    迷宫的模拟环境
    """
    action_dim = 4
    state_space_dim = 2

    def __init__(self, log_name='MazeEnv', maze=None):
        if maze is None:
            self.maze = Maze.build(bounds=(50, 50))
        else:
            self.maze = Maze(bounds=(maze.max_x, maze.max_y),
                             target=maze.target)

        # if MazeEnv.logger is None:
        #     MazeEnv.logger = Logger("MazeEnv")
        # self.logger = Logger(log_name, show_in_console=False)
        self.viewer = None
        # self.queue = Queue()

    def reset(self):
        x = random.randint(0, self.maze.max_x - 1)
        y = random.randint(0, self.maze.max_y - 1)

        self.maze = Maze.build(bounds=(self.maze.max_x, self.maze.max_y))

        if self.viewer is not None:
            self.viewer.set_maze(self.maze)

        return self.get_state()

    def step(self, a):
        """
        根据动作转换状态,返回新的状态(s), reward, done
        a: 0=up,1=down,2=left,3=right
        """
        # s = self.get_state()

        if a == 3:
            self.maze.move_up()
        if a == 2:
            self.maze.move_down()
        if a == 1:
            self.maze.move_left()
        if a == 0:
            self.maze.move_right()

        r = self.maze.snakes[0].delta_len() - 1
        done = False
        if self.maze.done():
            done = True
            r = -5

        return self.get_state(), r, done

    def render(self):
        if self.viewer is None:
            self.viewer = MazeViewer(self.maze)
        self.viewer.render()

    def get_state(self):
        return np.hstack([self.maze.snakes[0].x, self.maze.snakes[0].y])
