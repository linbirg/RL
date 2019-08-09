# import random
import pyglet

# from .maze import Maze
from env.maze import Maze

DimGrey = (69, 69, 69)  # 灰
LightSlateGray = (119, 136, 153)  # 	浅石板灰
Crimson = (220, 20, 60)  # DC143C 	猩红
RoyalBlue = (65, 105, 225)  # 皇家蓝
GhostWhite = (248, 248, 255)  # 幽灵白
Lavender = (230, 230, 250)  # 熏衣草花的淡紫色
AliceBlue = (240, 248, 255)  # 爱丽丝蓝
LightCoral = (240, 128, 128)  # 淡珊瑚色


class MazeViewer(pyglet.window.Window):

    OCCUPY = 90

    # pyglet.clock.ClockDisplay()

    def __init__(self, maze):
        super(MazeViewer, self).__init__(width=1000,
                                         height=800,
                                         resizable=False,
                                         caption='maze',
                                         vsync=False)

        self.maze = maze

        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.batch = pyglet.graphics.Batch()

        self.grids = self.plot_area()
        self.target = self.plot_target()
        self.bodys = self.plot_snakes()

    def set_maze(self, maze):
        self.maze = maze
        self.update()

    def render(self):
        # pyglet.clock.tick()
        self.update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def update(self):
        self._clear()

        self.grids = self.plot_area()
        self.target = self.plot_target()
        self.bodys = self.plot_snakes()

    def _clear(self):
        self.target.delete()
        for block in self.bodys:
            block.delete()

        for line in self.grids:
            line.delete()

    def _add_rect(self, pos, width, height, color=Lavender):
        # 添加蓝点
        rect = self.batch.add(
            4,
            pyglet.gl.GL_QUADS,
            None,  # 4 corners
            (
                'v2f',
                [
                    pos[0],
                    pos[1],  # x1, y1
                    pos[0],
                    pos[1] + height,  # x2, y2
                    pos[0] + width,
                    pos[1] + height,  # x3, y3
                    pos[0] + width,
                    pos[1]
                ]),  # x4, y4
            ('c3B', color * 4))  # color

        return rect

    def _add_v_line(self, pos, length, width=1, color=AliceBlue):
        vline = self._add_rect(pos, width, length, color=color)
        return vline

    def _add_h_line(self, pos, length, width=1, color=AliceBlue):
        hline = self._add_rect(pos, length, width, color=color)
        return hline

    def plot_area(self):
        w, h = self.get_size()
        b_w, b_h = self._calc_block_size()
        x, y = self._get_origin()
        lines = []
        for i in range(self.maze.max_x + 1):
            vline = self._add_v_line((x + i * b_w, y), h * self._get_occupy())
            lines.append(vline)

        for i in range(self.maze.max_y + 1):
            hline = self._add_h_line((x, y + i * b_h), w * self._get_occupy())
            lines.append(hline)

        return lines

    def plot_target(self):
        x, y = self._get_origin()
        b_w, b_h = self._calc_block_size()
        target = self._add_rect(
            (x + self.maze.target[0] * b_w, y + self.maze.target[1] * b_h),
            b_w,
            b_h,
            color=LightCoral)

        return target

    def plot_snakes(self):
        if self.maze.snakes is None:
            return None

        bodys = []
        heads = []
        for snake in self.maze.snakes:
            bodys = bodys + snake.body[0:-1]
            heads.append(snake.body[-1])

        x, y = self._get_origin()
        b_w, b_h = self._calc_block_size()
        blocks = []
        for block in bodys:
            blck_rect = self._add_rect(
                (x + block[0] * b_w, y + block[1] * b_h),
                b_w,
                b_h,
                color=RoyalBlue)
            blocks.append(blck_rect)

        for head in heads:
            blck_rect = self._add_rect((x + head[0] * b_w, y + head[1] * b_h),
                                       b_w,
                                       b_h,
                                       color=Crimson)
            blocks.append(blck_rect)

        return blocks

    def _get_origin(self):
        w, h = self.get_size()
        return w * (1 - self._get_occupy()) / 2, h * (1 -
                                                      self._get_occupy()) / 2

    def _get_occupy(self):
        return self.OCCUPY / 100

    def _calc_block_size(self):
        w, h = self.get_size()
        area_w = w * self._get_occupy()
        area_h = h * self._get_occupy()

        block_w = area_w / self.maze.max_x
        block_h = area_h / self.maze.max_y

        return (block_w, block_h)


def viewer_run():
    viewer = MazeViewer(Maze())
    while True:
        viewer.render()


if __name__ == '__main__':
    import time
    import random
    import logging
    maze = Maze.build((10, 10))
    viewer = MazeViewer(maze)
    while True:
        dx, dy = random.randint(-1, 1), random.randint(-1, 1)
        print(dx, dy)
        maze.move(dx, dy)
        viewer.render()
        time.sleep(0.1)
