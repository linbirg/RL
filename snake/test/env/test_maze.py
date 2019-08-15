#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Author: yizr

import os
import sys

__abs_file__ = os.path.abspath(__file__)
env_dir = os.path.dirname(__abs_file__)
test_dir = os.path.dirname(env_dir)
code_dir = os.path.dirname(test_dir)
sys.path.append(code_dir)

from env.maze import Snake

import unittest


class TestMazeCase(unittest.TestCase):
    def test_bodys(self):
        print('test')
        snake = Snake()
        snake.move(1, 0)
        snake.move(0, 1)
        snake.eat((1, 2))
        print(snake.bodys)


if __name__ == "__main__":
    unittest.main()
