from multiprocessing import Process
from kchart import KChart

import load_data as ld
import date_time_util as dtutil

# 进程版的kchart
class KChartProcess(Process):
    def __init__(self, queue,data_file="GBPUSD15.csv"):
        Process.__init__(self)
        self.queue = queue

        self.data_file = data_file
        self.data = self._get_ohlc()

    def run(self):
        self.kchart = KChart(self.data)
        self.kchart.animate(self.queue)
    
    def _get_ohlc(self):
        ohlcs = ld.load_cvs(self.data_file)
        # [[t,open,high,low,close]]
        prices = [[
            dtutil.date_time_str_2_float(olhc[0] + ' ' + olhc[1]),
            float(olhc[2]),
            float(olhc[3]),
            float(olhc[4]),
            float(olhc[5])
        ] for olhc in ohlcs]
        return prices

if __name__ == "__main__":
    from multiprocessing import Queue
    import time
    queue = Queue()
    p = KChartProcess(queue)
    p.start()
    step = 0
    while True:
        queue.put({"step":step, "acct":0, "positions":[]})
        step += 1
        time.sleep(0.5)
