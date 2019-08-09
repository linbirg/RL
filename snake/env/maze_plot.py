"""
迷宫的画图程序，暂时用plot画
"""
from matplotlib import patches as patch

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from maze import Maze
from logger import Logger


class MazePlot(Maze):
    def __init__(self, maze):
        super(MazePlot, self).__init__(
            start=(maze.x, maze.y),
            bounds=(maze.max_x, maze.max_y),
            door=maze.door,
            blocks=maze.blocks)
        self.logger = Logger('MazePlot', show_in_console=False)

        self.circle = None

        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        self.ax.autoscale_view()
        self.ax.grid(True)
        self.plot_area()
        self.plot_grid()
        self.plot_door()
        self.plot_blocks()
        self.anim = None
        self.queue = None

    def plot_area(self):
        self.ax.set_xlim(0, self.max_x)
        self.ax.set_ylim(0, self.max_y)

    def plot_grid(self):
        for i in range(self.max_x):
            self.plot_line(start=(i + 1, 0), end=(i + 1, self.max_y))
        for i in range(self.max_y):
            self.plot_line(start=(0, i + 1), end=(self.max_x, i + 1))

    def plot_line(self, start, end, color='black'):
        vline = Line2D(
            xdata=(start[0], end[0]),
            ydata=(start[1], end[1]),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )
        self.ax.add_line(vline)

    def plot_current(self):
        if self.circle is None:
            self.circle = patch.Circle(
                xy=(self.x + 0.5, self.y + 0.5),
                radius=0.3,
                alpha=0.5,
                color='red')
            self.ax.add_patch(self.circle)
        else:
            self.circle.center = (self.x + 0.5, self.y + 0.5)

    def plot_door(self):
        self._plot_rect(self.door)

    def _plot_rect(self, xy, color='b'):
        rect = Rectangle(
            xy=xy, width=1, height=1, facecolor=color, edgecolor=color)
        rect.set_alpha(1)
        self.ax.add_patch(rect)

    def plot_blocks(self):
        if self.blocks is not None:
            for block in self.blocks:
                self._plot_rect(block, 'k')

    def animate_init_func(self):
        if self.circle is None:
            self.plot_current()

        return self.circle,

    def update_plot(self, i):
        self.move()
        self.logger.debug((self.x, self.y))
        self.plot_current()
        return self.circle,

    def move(self):
        cmd = self.queue.get()
        if cmd['cmd'] == 'reset':
            self.x = cmd['xy'][0]
            self.y = cmd['xy'][1]

        elif cmd['cmd'] == 'move':
            a = cmd['move']
            if a == 0:
                self.move_up()
            if a == 1:
                self.move_down()
            if a == 2:
                self.move_left()
            if a == 3:
                self.move_right()

    def animate(self, que):
        self.queue = que
        self.anim = animation.FuncAnimation(
            fig=self.fig,
            init_func=self.animate_init_func,
            func=self.update_plot,
            interval=100,
            frames=10,
            repeat=True,
            blit=True)

        plt.show()


# if __name__ == "__main__":
#     import time
#     import random
#     import multiprocessing as mlp
#     maze = MazePlot()
#     queue = mlp.Queue()
#     maze.animate(queue)
#     while True:
#         queue.put(random.randint(0, 3))
#         time.sleep(1)
