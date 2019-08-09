"""
迷宫的模拟环境,只负责迷宫的基本属性和动作，不负责绘图的部分
"""
import random


class Snake:
    def __init__(self, head=(0, 0)):
        super().__init__()

        self.init(head)

    def init(self, head=(0, 0)):
        self.body = [head]
        self.pre_len = 1

    def move(self, dx, dy):
        head = self.body[-1]
        head = head[0] + dx, head[1] + dy
        self.body.append(head)
        self.body.remove(self.body[0])

    def up(self):
        self.move(0, 1)

    def down(self):
        self.move(0, -1)

    def left(self):
        self.move(-1, 0)

    def right(self):
        self.move(1, 0)

    def eat(self, p):
        self.body.append(p)

    def length(self):
        return len(self.body)

    def delta_len(self):
        return len(self.body) - self.pre_len

    def on_body(self, x, y):
        for p in self.body:
            if x == p[0] and y == p[1]:
                return True

        return False

    @property
    def x(self):
        return self.body[-1][0]

    @property
    def y(self):
        return self.body[-1][1]


class Maze(object):
    """
    贪吃蛇的模拟环境
    """
    def __init__(self, bounds=(5, 5), target=(4, 4), start=None):
        self.max_y = bounds[0]
        self.max_x = bounds[1]
        self.target = target
        if start is None:
            start = (random.randint(0, bounds[0] - 1),
                     random.randint(0, bounds[1] - 1))

        self.snakes = [Snake(start)]
        self.is_done = False

    def is_target(self, x, y):
        return True if x == self.target[0] and y == self.target[1] else False

    def on_snakes(self, x, y):
        if self.snakes is None or len(self.snakes) == 0:
            return False

        for snake in self.snakes:
            if snake.on_body(x, y):
                return True

        return False

    def is_in_area(self, x, y):
        return True if x >= 0 and x < self.max_x and y >= 0 and y < self.max_y else False

    def is_safe_block(self, x, y):
        return True if self.is_in_area(
            x, y) and (not self.on_snakes(x, y)) else False

    def set_target(self, target):
        if self.is_safe_block(target[0], target[1]):
            self.target = target
            return True

        return False

    def random_target(self):
        target = (random.randint(0, self.max_x - 1),
                  random.randint(0, self.max_y - 1))
        if not self.set_target(target):
            self.random_target()

    def move(self, dx, dy, index=0):
        if self.is_target(self.snakes[index].x + dx,
                          self.snakes[index].y + dy):
            self.snakes[index].eat(self.target)
            self.random_target()
            return True

        if self.is_safe_block(self.snakes[index].x + dx,
                              self.snakes[index].y + dy):
            self.snakes[index].move(dx, dy)
            return True

        self.is_done = True
        return False

    def move_up(self, index=0):
        return self.move(0, 1, index)

    def move_down(self, index=0):
        return self.move(0, -1, index)

    def move_left(self, index=0):
        return self.move(-1, 0, index)

    def move_right(self, index=0):
        return self.move(1, 0, index)

    def done(self, index=0):
        return self.is_done

    @classmethod
    def build(cls, bounds=(5, 5)):
        target = (random.randint(0, bounds[0] - 1),
                  random.randint(0, bounds[1] - 1))

        head = (random.randint(0, bounds[0] - 1),
                random.randint(0, bounds[1] - 1))

        tmp = Maze(bounds=bounds, target=head)

        return tmp

    # def add_block(self, block):
    #     if self.blocks is None:
    #         self.blocks = [block]
    #         return

    #     if self.is_safe_block(block[0], block[1]) and not self.is_door(
    #             block[0], block[1]):  # 添加的时候不能是出口
    #         self.blocks.append(block)


# if __name__ == "__main__":
#     maze = Maze()
