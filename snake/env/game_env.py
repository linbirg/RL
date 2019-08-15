"""
迷宫的模拟环境
"""
import random
import numpy as np
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
        # if maze is None:
        #     self.maze = Game(bounds=(50, 50))
        # else:
        self.maze = Game(bounds=(maze.max_x, maze.max_y))

        self.viewer = None

    def reset(self):
        # x = random.randint(0, self.maze.max_x - 1)
        # y = random.randint(0, self.maze.max_y - 1)

        self.maze = Game(bounds=(self.maze.max_x, self.maze.max_y))

        if self.viewer is not None:
            self.viewer.set_maze(self.maze)

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

        r = self.maze.snake.delta_len() * 2 - 0.1

        done = False
        if self.maze.done():
            done = True
            r = -1

        return self.get_state(), r, done

    def render(self):
        if self.viewer is None:
            self.viewer = MazeViewer(self.maze)
        self.viewer.render()

    def get_state(self):
        return np.hstack(
            [self.maze.target[0], self.maze.target[1], *self.maze.snake.bodys])


class SnakeGameEnvTriAct(SnakeGameEnv):
    action_dim = 3

    def __init__(self, log_name='MazeEnv', game=None):
        super().__init__(log_name, game)

    def step(self, a):
        """
        根据动作转换状态,返回新的状态(s), reward, done
        a: 0=up,1=down,2=left,3=right
        """
        if a == 2:
            self.maze.move_down()
        if a == 1:
            self.maze.move_left()
        if a == 0:
            self.maze.move_right()

        r = self.maze.snake.delta_len() * 2 - 0.1

        done = False
        if self.maze.done():
            done = True
            r = -1

        return self.get_state(), r, done
