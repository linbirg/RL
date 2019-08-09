import plot as pl
import load_data as ld
import date_time_util as dtutil
from matplotlib.dates import date2num
import sched
import time

import matplotlib.pyplot as plt  
import matplotlib.finance as mpf 
import matplotlib.animation as animation



DATA_FILE = "GBPUSD15.csv"


def get_ohlc():
    ohlcs = ld.load_cvs(DATA_FILE)
    prices = [(
        dtutil.date_time_str_2_float(olhc[0]+' '+olhc[1]),
        float(olhc[2]),
        float(olhc[3]),
        float(olhc[4]),
        float(olhc[5])
     ) for olhc in ohlcs]
    return prices

# N = 100
# start = 0

# def plot_candle(quotes):
#     global start
#     pl.plot_candle(quotes[start:start+N])
#     start += 1

quotes = get_ohlc()
N = 100
start = 0
fig, ax = plt.subplots(facecolor=(0.5, 0.5, 0.5)) 
lines = []
patchs = []

def init_func():
    global fig
    global ax

    fig, ax = plt.subplots(facecolor=(0.5, 0.5, 0.5)) 
    return fig.canvas

def update(i):
    global quotes
    global N
    global start
    global ax
    global candle
    global lines
    global patchs
    lines,patchs = mpf.candlestick_ohlc(ax,quotes[start:start+N],width=240,colorup='r',colordown='green')
    start += 1
    print(start)
    return fig.canvas

def test_animation():
    global fig
    global ax
    fig.subplots_adjust(bottom=0.2)  
    # 设置X轴刻度为日期时间  
    # ax.xaxis_date()
    ax.autoscale_view() 
    # X轴刻度文字倾斜45度  
    plt.xticks(rotation=45)
    plt.title(u"601558")
    plt.xlabel(u"time")
    plt.ylabel(u"price")
    ani = animation.FuncAnimation(fig, update, interval=25,blit=True)
    # mpf.candlestick_ohlc(ax,quotes,width=240,colorup='r',colordown='green')  
    plt.grid(True)
    plt.show()


def delete_lin(i):
    global quotes


def test_remove_lines():
    global fig
    global ax
    
    
    plt.grid(True)
    plt.show()
    
if __name__ == '__main__': 
    test_remove_lines()
    