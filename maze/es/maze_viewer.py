# import random
import pyglet

from maze import Maze

DimGrey = (69, 69, 69)
LightSlateGray = (119, 136, 153)
Crimson = (220, 20, 60)  # DC143C


class MazeViewer(pyglet.window.Window):

    OCCUPY = 80

    # pyglet.clock.ClockDisplay()

    def __init__(self, maze):
        super(MazeViewer, self).__init__(
            width=1000,
            height=800,
            resizable=False,
            caption='maze',
            vsync=False)

        self.maze = maze

        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.batch = pyglet.graphics.Batch()

        self.grids = self.plot_area()
        self.door = self.plot_door()
        self.blocks = self.plot_blocks()
        self.cur = self.plot_cur()

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

        self.plot_area()
        self.door = self.plot_door()
        self.blocks = self.plot_blocks()
        self.cur = self.plot_cur()

    def _clear(self):
        self.cur.delete()
        self.door.delete()
        for block in self.blocks:
            block.delete()

        for line in self.grids:
            line.delete()

    def _add_rect(self, pos, width, height, color=(80, 60, 255)):
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

    def _add_v_line(self, pos, length, width=1, color=DimGrey):
        vline = self._add_rect(pos, width, length, color=color)
        return vline

    def _add_h_line(self, pos, length, width=1, color=DimGrey):
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

    def plot_door(self):
        x, y = self._get_origin()
        b_w, b_h = self._calc_block_size()
        door = self._add_rect(
            (x + self.maze.door[0] * b_w, y + self.maze.door[1] * b_h), b_w,
            b_h)

        return door

    def plot_cur(self):
        x, y = self._get_origin()
        b_w, b_h = self._calc_block_size()
        cur = self._add_rect(
            (x + self.maze.x * b_w, y + self.maze.y * b_h),
            b_w,
            b_h,
            color=Crimson)

        return cur

    def plot_blocks(self):
        if self.maze.blocks is None:
            return None

        x, y = self._get_origin()
        b_w, b_h = self._calc_block_size()
        blocks = []
        for block in self.maze.blocks:
            blck_rect = self._add_rect(
                (x + block[0] * b_w, y + block[1] * b_h),
                b_w,
                b_h,
                color=LightSlateGray)
            blocks.append(blck_rect)

        return blocks

    def _get_origin(self):
        w, h = self.get_size()
        return w * (1 - self._get_occupy()) / 2, h * (
            1 - self._get_occupy()) / 2

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
    viewer = MazeViewer(
        Maze(blocks=[(1, 1), (1, 2), (1, 3), (2, 1), (3, 1), (4, 2)]))
    while True:
        viewer.render()


if __name__ == '__main__':
    import threading
    import time
    t = threading.Thread(target=viewer_run)
    t.start()
    while True:
        time.sleep(0.5)
