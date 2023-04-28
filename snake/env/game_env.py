"""
迷宫的模拟环境
"""
import random
import numpy as np
import math
# from logger import Logger

from env.game import GameTriAct, Game
from env.maze_viewer import MazeViewer

# from process_maze import ProcessMaze


class SnakeGameEnv(object):
    """
    迷宫的模拟环境
    """
    action_dim = 4
    state_space_dim = 18

    def __init__(self, maze, log_name='MazeEnv'):
        self.maze = maze

        self.viewer = None
        self.womiga = self.calc_distance()

    def reset(self):
        self.maze = Game(bounds=(self.maze.max_x, self.maze.max_y))

        if self.viewer is not None:
            self.viewer.set_maze(self.maze)

        self.womiga = self.calc_distance()

        return self.get_state()

    def step(self, a):
        """
        根据动作转换状态,返回新的状态(s), reward, done
        a: 0=up,1=down,2=left,3=right
        """

        if a == 3:
            self.maze.move_up()
        if a == 2:
            self.maze.move_down()
        if a == 1:
            self.maze.move_left()
        if a == 0:
            self.maze.move_right()

        r = self.maze.snake.length() + 0.9 * self.get_womiga_reword() - 0.5

        done = False
        if self.maze.done():
            done = True
            r = -1.5

        return self.get_state(), r, done

    def render(self):
        if self.viewer is None:
            self.viewer = MazeViewer(self.maze)
        self.viewer.render()

    def get_state(self):
        return np.hstack(
            [self.maze.target[0], self.maze.target[1], *self.maze.snake.bodys])

    def calc_distance(self):
        x, y = self.maze.target
        hx, hy = self.maze.snake.x, self.maze.snake.y

        return math.sqrt((x - hx) * (x - hx) + (y - hy) * (y - hy))

    def get_womiga_reword(self):
        womiga = self.calc_distance()
        delta_w = self.womiga - womiga
        self.womiga = womiga
        return delta_w


class SnakeGameEnvTriAct(SnakeGameEnv):
    action_dim = 3

    def __init__(self, game, log_name='MazeEnv'):
        super().__init__(game, log_name)

    def step(self, a):
        """
        根据动作转换状态,返回新的状态(s), reward, done
        a: 0=up,1=left,2=right
        """
        if a == 0:
            self.maze.forward()
        if a == 1:
            self.maze.left()
        if a == 2:
            self.maze.right()

        r = self.maze.snake.length() + 0.5 * self.get_womiga_reword() - 0.9

        done = False
        if self.maze.done():
            done = True
            r = -1.5

        return self.get_state(), r, done

    def reset(self):
        self.maze = GameTriAct(bounds=(self.maze.max_x, self.maze.max_y))

        if self.viewer is not None:
            self.viewer.set_maze(self.maze)

        self.womiga = self.calc_distance()

        return self.get_state()
