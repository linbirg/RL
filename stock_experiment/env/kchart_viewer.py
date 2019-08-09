import pyglet


class KChartViewer(pyglet.window.Window):
    OCCUPY = 80
    # color={
    DimGrey = (69, 69, 69)
    LightSlateGray = (119, 136, 153)
    Crimson = (220, 20, 60)

    # }

    def __init__(self, xdim, ydim):
        super(KChartViewer, self).__init__(
            width=1000,
            height=800,
            resizable=False,
            caption='kchart',
            vsync=False)

        self.xdim = xdim
        self.ydim = ydim
        self.y_grids_cnt = 10
        self.x_grids_cnt = 10

        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.batch = pyglet.graphics.Batch()  # 所有的对象都添加在batch中

        self.grids = self.plot_grids()

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
        # self._clear()

        self.grids = self.plot_grids()

    def plot_grids(self):
        w, h = self.get_size()
        x, y = self._get_origin()
        b_w, b_h = self._calc_block_size()
        lines = []
        for i in range(self.x_grids_cnt):
            vline = self._add_v_line((x + i * b_w, y), h * self._get_occupy())
            lines.append(vline)

        for i in range(self.y_grids_cnt):
            hline = self._add_h_line((x, y + i * b_h), w * self._get_occupy())
            lines.append(hline)

        return lines

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

        block_w = area_w / (self.x_grids_cnt - 1)
        block_h = area_h / (self.y_grids_cnt - 1)

        return (block_w, block_h)


if __name__ == '__main__':
    import time
    kchart = KChartViewer((0, 5), (1, 10))
    while True:
        kchart.render()
        time.sleep(0.5)
