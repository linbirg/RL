"""
Environment for stock simulation
"""

from multiprocessing import Queue

from process_kchart import KChartProcess
import load_data as ld
import date_time_util as dtutil

from simulator import Simulator

from logger import Logger
# import list_to_str as l2s

import numpy as np


class StockEnv(object):
    logger = None
    simulator = None
    data = None
    # render_process = None
    queue = Queue()

    action_bound = [0, 2]

    data_path = "GBPUSD15.csv"
    positions = []  # (t,price) 啥时候，以什么价格开的仓(暂时只买)
    max_positions = 10
    # acct = 20000  # 资金
    # profit = 0  # 收益 
    # holdings = 0  # 持仓价值

    # kChart = None

    points = 10000  # 价格的小数点位数，精度
    multi = 2  # 杠杆倍数

    def __init__(self):
        """ docs"""
        self.data = self._get_ohlc()
        self.simulator = Simulator(self.data)
        self.queue = Queue()
        if StockEnv.logger is None:
            StockEnv.logger = Logger("StockEnv")
        self.acct = 20000
        self.profit = 0  # 收益
        self.holdings = 0  # 持仓价值
        self._reward = Reward(self.acct)
        self.kChart = None
        self.total = 0
        self.occupy = 0

    def reset(self):
        self.simulator = Simulator(self.data)
        self.positions = []
        self.acct = 20000
        self.profit = 0
        self.holdings = 0
        self.total = 0
        self.kChart = None
        self.clear_queue()
        self._reward = Reward(self.acct)
        return self._get_state()

    def clear_queue(self):
        while not self.queue.empty():
            self.queue.get()

    def step(self, a):
        # 0 啥也不做 1 开买仓 2开卖仓  3 平仓
        if a == 0:
            pass
            # 计算收益
        if a == 1:
            # 开一手买，扣减手续费，计算收益
            self._open('b')
        if a == 2:
            # 开卖
            self._open('s')
        if a == 3:
            self._close()

        self.profit, self.holdings, self.total = self._calc_profit()
        # if not self.kChart is None:
        #     # msg = l2s.format_to_str([a, 'acct:', self.acct, 'positions:',self.positions])
        #     self.logger.debug([a, 'total:', self.acct+self.profit, 'positions:',self.positions])

        done = self.simulator.done() or (self.total <= 0)
        rd = self._reward.get_rd(self.acct + self.occupy)
        if self.total <= 0:
            rd = -10
        self.next_step()
        return self._get_state(), rd, done

    def render(self):
        if self.kChart is None:
            self.kChart = KChartProcess(self.queue, self.data_path)
            self.kChart.start()

    def get_state(self):
        return self._get_state()

    def _get_ohlc(self):
        ohlcs = ld.load_cvs(self.data_path)
        # [[t,open,high,low,close]]
        prices = [[
            dtutil.date_time_str_2_float(olhc[0] + ' ' + olhc[1]),
            float(olhc[2]),
            float(olhc[3]),
            float(olhc[4]),
            float(olhc[5])
        ] for olhc in ohlcs]
        return prices

    def _get_state(self):
        return np.hstack([self.simulator.latest, len(self.positions)])

    def _calc_profit(self):
        price = self._get_latest_ohlc()
        sum_profit = 0
        holdings = 0
        positions = []
        pos_profit = 0
        for position in self.positions:
            holdings += price[1] * self.points
            pos_profit = (price[1] - position[1]) * self.points
            if position[2] == 's':
                pos_profit = -1 * pos_profit

            sum_profit += pos_profit

            tmp_p = (position[0], position[1], position[2], position[3],
                     pos_profit)
            positions.append(tmp_p)

        self.positions = positions  # 更新持仓收益

        return sum_profit, holdings, self.acct + self.occupy + sum_profit

    # 以开盘价作为最新价
    def _get_latest_ohlc(self):
        return self.simulator.latest

    def _open(self, buy_or_sell):
        if len(self.positions) < self.max_positions:
            ohlc = self._get_latest_ohlc()
            amount = ohlc[1] * self.points / self.multi
            if self.acct >= amount + 5:  # 手续费固定5块钱
                self.positions.append((ohlc[0], ohlc[1], buy_or_sell, amount,
                                       0))
                self.acct -= amount + 5
                self.occupy += amount
                # self.logger.debug(["s",self._get_state(),'a','open','acct',self.acct,'positions',self.positions])

    def _close(self):
        cnt = len(self.positions)
        if cnt > 0:  # 平最老的仓，计算收益
            position = self.positions[0]
            self.positions = self.positions[1:]
            price = self._get_latest_ohlc()[1]  # 开盘价
            factor = 1 if position[2] == 'b' else -1
            amount = position[3]
            self.occupy -= amount
            self.acct += amount
            self.acct += factor * (price - position[1]) * self.points

    def next_step(self):
        self.simulator.next()
        # 每走一个tick,钱按一定比例贬值
        self.devalue()
        if self.kChart is not None:
            self.queue.put({
                "step": self.simulator.step,
                "total": self.total,
                "positions": self.positions
            })

    def devalue(self):
        # 让钱按照一定比例贬值
        self.acct = self.acct * (0.99999)


class Reward(object):
    """docs"""

    def __init__(self, total=1):
        self.pre = total
        self.cur = self.pre

    def get_rd(self, value):
        self.pre = self.cur
        self.cur = value
        if self.cur < 1:
            self.cur = 1
        return self.cur - self.pre
