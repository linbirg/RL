"""
迷宫的模拟环境,只负责迷宫的基本属性和动作，不负责绘图的部分
"""
import random


class Maze(object):
    """
    迷宫的模拟环境
    """

    def __init__(self, start=(0, 0), bounds=(5, 5), door=(4, 4), blocks=None):
        # 定义5*5的迷宫
        self.max_y = bounds[0]
        self.max_x = bounds[1]
        # 定义门，如果移动到门，则胜利
        self.door = door
        self.blocks = blocks

        # 初始位置
        self.x = start[0]
        self.y = start[1]

    def is_door(self, x, y):
        return True if x == self.door[0] and y == self.door[1] else False

    def is_block(self, x, y):
        if self.blocks is None:
            return False

        for block in self.blocks:
            if x == block[0] and y == block[1]:
                return True
        # 不等于所有的block
        return False

    def is_in_area(self, x, y):
        return True if x >= 0 and x < self.max_x and y >= 0 and y < self.max_y else False

    def is_safe_block(self, x, y):
        return True if self.is_in_area(
            x, y) and (not self.is_block(x, y)) else False

    def set_blocks(self, blocks):
        self.blocks = blocks

    def set_start(self, start):
        if self.is_safe_block(start[0], start[1]):  #
            self.x = start[0]
            self.y = start[1]
            return True

        return False

    def set_door(self, door):
        if self.is_safe_block(door[0], door[1]):
            self.door = door
            return True

        return False

    def move_up(self):
        if self.is_safe_block(self.x, self.y + 1):
            self.y = self.y + 1
            return True
        return False

    def move_down(self):
        if self.is_safe_block(self.x, self.y - 1):
            self.y = self.y - 1
            return True

        return False

    def move_left(self):
        if self.is_safe_block(self.x - 1, self.y):
            self.x = self.x - 1
            return True

        return False

    def move_right(self):
        if self.is_safe_block(self.x + 1, self.y):
            self.x += 1
            return True

        return False

    def done(self):
        return True if self.is_door(self.x, self.y) else False

    @classmethod
    def build(cls, bounds=(5, 5), door=None, blocks=None, block_cnt=None):
        """
        创建迷宫,bounds 指定长宽,door指定出口，如果不指定，则随机生成。blocks，如果不指定，也不指定block_cnt，则取总格子的40%，如果指定block_cnt，则随机生成指定数量的block。
        """
        if door is None:
            door = (random.randint(0, bounds[0] - 1),
                    random.randint(0, bounds[1] - 1))

        tmp = Maze(bounds=bounds, door=door, blocks=blocks)
        if blocks is None:
            if block_cnt is None or block_cnt <= 0:
                block_cnt = (int)(
                    bounds[0] * bounds[1] * 40 / 100)  # block占40%

            for i in range(block_cnt):
                block = (random.randint(0, tmp.max_x - 1),
                         random.randint(0, tmp.max_y - 1))
                tmp.add_block(block)
        return tmp

    def add_block(self, block):
        if self.blocks is None:
            self.blocks = [block]
            return

        if self.is_safe_block(block[0], block[1]) and not self.is_door(
                block[0], block[1]):  # 添加的时候不能是出口
            self.blocks.append(block)


if __name__ == "__main__":
    maze = Maze(blocks=[(1, 1), (1, 2), (1, 3), (2, 1), (3, 1)])

    def test_is_door():
        print(maze.is_door(3, 4))
        print(maze.is_door(4, 4))
        print(maze.is_door(5, 4))
        print(maze.is_door(0, 2))

    def test_is_block():
        print("test_is_block")
        print(maze.is_block(0, 0))
        print(maze.is_block(0, 1))
        print(maze.is_block(1, 1))
        print(maze.is_block(2, 1))
        print(maze.is_block(2, 2))
        print(maze.is_block(3, 2))
        print(maze.is_block(3, 1))

    def test_is_safe_block():
        print("test_is_safe_block")
        print(maze.is_safe_block(0, 0))
        print(maze.is_safe_block(0, 1))
        print(maze.is_safe_block(1, 1))
        print(maze.is_safe_block(5, 5))
        print(maze.is_safe_block(4, 6))

    def test_move_to_door():
        print("test_move_to_door")
        print("s", (maze.x, maze.y))
        print("up", maze.move_up())
        print("up", maze.move_up())
        print("up", maze.move_up())
        print("up", maze.move_up())
        print("right", maze.move_right())
        print("right", maze.move_right())
        print("right", maze.move_right())
        print("right", maze.move_right())
        print("done", maze.done())

    test_is_door()
    test_is_block()
    test_is_safe_block()
    test_move_to_door()
