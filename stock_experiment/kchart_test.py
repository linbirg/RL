from kchart import KChart
import load_data as ld
import date_time_util as dtutil
import time
import multiprocessing as mlp
from multiprocessing import Process

DATA_FILE = "GBPUSD15.csv"


def get_ohlc():
    ohlcs = ld.load_cvs(DATA_FILE)
    prices = [[
        dtutil.date_time_str_2_float(olhc[0]+' '+olhc[1]),
        float(olhc[2]),
        float(olhc[3]),
        float(olhc[4]),
        float(olhc[5])
     ] for olhc in ohlcs]
    return prices

# chart = None

# def animate():
#     global chart
#     chart.animate()


if __name__ == "__main__":
    chart = KChart(data=get_ohlc())
    queue = mlp.Queue()
    p = Process(target=chart.animate, args=(queue,))
    p.start()
    step = 0
    while True:
        queue.put({"step":step, "acct":0, "positions":[]})
        step += 1
        time.sleep(0.5)