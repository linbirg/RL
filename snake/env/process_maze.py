"""
进程版的maze画图程序
"""

from multiprocessing import Process
from maze import Maze
from maze_plot import MazePlot


class ProcessMaze(Process):
    def __init__(self, que, maze=None):
        Process.__init__(self)
        self.queue = que
        self.maze = maze

    def run(self):
        if self.maze is None:
            self.maze = Maze()
        self.maze_plot = MazePlot(self.maze)
        self.maze_plot.animate(self.queue)


if __name__ == "__main__":
    from multiprocessing import Queue
    import time
    import random
    queue = Queue()
    p = ProcessMaze(queue)
    p.start()
    while True:
        queue.put(random.randint(0, 3))
        time.sleep(0.5)
