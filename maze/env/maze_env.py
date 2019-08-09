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
            self.maze = Maze.build(bounds=(10, 10), block_cnt=20)
        else:
            self.maze = Maze(
                start=(maze.x, maze.y),
                bounds=(maze.max_x, maze.max_y),
                door=maze.door,
                blocks=maze.blocks)

        # if MazeEnv.logger is None:
        #     MazeEnv.logger = Logger("MazeEnv")
        # self.logger = Logger(log_name, show_in_console=False)
        self.viewer = None
        # self.queue = Queue()

    def reset(self):
        x = random.randint(0, self.maze.max_x - 1)
        y = random.randint(0, self.maze.max_y - 1)
        # x, y = 0, 0
        self.maze.set_start((x, y))
        # self.maze = Maze.build(bounds=(20, 20), block_cnt=100)

        if self.viewer is not None:
            self.viewer.maze = self.maze
        return self.get_state()

    # def clear_queue(self):
    #     while not self.queue.empty():
    #         self.queue.get()

    def step(self, a):
        """
        根据动作转换状态,返回新的状态(s), reward, done
        a: 0=up,1=down,2=left,3=right
        """
        # s = self.get_state()
        succ = False

        if a == 3:
            succ = self.maze.move_up()
        if a == 2:
            succ = self.maze.move_down()
        if a == 1:
            succ = self.maze.move_left()
        if a == 0:
            succ = self.maze.move_right()

        r = 0  # 每走一步-1分，直到门，相当于策略要用最短的步数走出去
        if not succ:  # 对撞墙等错误行为惩罚
            r = 0
        done = False
        if self.maze.done():
            done = True
            r = 10
        # self.logger.debug([s, a, self.get_state(), r, done])
        # if self.viewer is not None:
        #     self.viewer.maze.set_start(start=(self.maze.x, self.maze.y))

        return self.get_state(), r, done

    def render(self):
        if self.viewer is None:
            self.viewer = MazeViewer(self.maze)
        self.viewer.render()

    def get_state(self):
        return np.hstack([self.maze.x, self.maze.y])
