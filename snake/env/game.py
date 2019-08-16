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

    def is_back(self, x, y):
        if len(self.body) < 2:
            return False

        sec = self.body[-2]
        return True if x == sec[0] and y == sec[1] else False

    @property
    def x(self):
        return self.body[-1][0]

    @property
    def y(self):
        return self.body[-1][1]

    @property
    def tail(self):
        return self.body[0]

    @property
    def bodys(self):
        bodys = [-1, -1] * 8
        cnt = min(8, self.length())
        for i in range(cnt):
            bodys[i * 2] = self.body[i][0]
            bodys[i * 2 + 1] = self.body[i][1]

        return bodys

    def is_up(self):
        assert self.length() > 1
        head = self.body[-1]
        neck = self.body[-2]

        return head[1] == neck[1] and head[0] > neck[0]

    def is_down(self):
        assert self.length() > 1
        head = self.body[-1]
        neck = self.body[-2]

        return head[1] == neck[1] and head[0] < neck[0]

    def is_left(self):
        assert self.length() > 1
        head = self.body[-1]
        neck = self.body[-2]

        return head[0] == neck[0] and head[1] < neck[1]

    def is_right(self):
        assert self.length() > 1
        head = self.body[-1]
        neck = self.body[-2]

        return head[0] == neck[0] and head[1] > neck[1]


class Game(object):
    """
    贪吃蛇的模拟环境
    """
    def __init__(self, bounds=(5, 5)):
        self.max_y = bounds[0]
        self.max_x = bounds[1]
        self.is_done = False
        self.snake = Snake(self._random_pos())

        self.random_target()

    def _random_pos(self):
        return (random.randint(0, self.max_x - 1),
                random.randint(0, self.max_y - 1))

    def is_target(self, x, y):
        return True if x == self.target[0] and y == self.target[1] else False

    def on_snakes(self, x, y):
        return self.snake.on_body(x, y)

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
        target = self._random_pos()
        if not self.set_target(target):
            self.random_target()

    def move(self, dx, dy):
        if self.is_target(self.snake.x + dx, self.snake.y + dy):
            self.snake.eat(self.target)

            self.random_target()
            return True

        if self.snake.is_back(self.snake.x + dx, self.snake.y + dy):
            return False

        if self.is_safe_block(self.snake.x + dx, self.snake.y + dy):
            self.snake.move(dx, dy)
            return True

        self.is_done = True
        return False

    def move_up(self):
        return self.move(0, 1)

    def move_down(self, index=0):
        return self.move(0, -1)

    def move_left(self):
        return self.move(-1, 0)

    def move_right(self):
        return self.move(1, 0)

    def done(self):
        return self.is_done


class GameTriAct(Game):
    def __init__(self, bounds=(5, 5)):
        return super().__init__(bounds=bounds)

    def is_down(self):
        pass

    def forward(self):
        if self.snake.length() == 1:
            # 如果长度1，向前固定向上
            return self.move_up()

        if self.snake.is_down():
            return self.move_down()

        if self.snake.is_up():
            return self.move_up()

        if self.snake.is_left():
            return self.move_left()

        if self.snake.is_right():
            return self.move_right()

    def left(self):
        if self.snake.length() == 1:
            return self.move_left()

        if self.snake.is_down():
            return self.move_right()

        if self.snake.is_up():
            return self.move_left()

        if self.snake.is_left():
            return self.move_down()

        if self.snake.is_right():
            return self.move_up()

    def right(self):
        if self.snake.length() == 1:
            return self.move_right()

        if self.snake.is_down():
            return self.move_left()

        if self.snake.is_up():
            return self.move_right()

        if self.snake.is_left():
            return self.move_up()

        if self.snake.is_right():
            return self.move_down()
