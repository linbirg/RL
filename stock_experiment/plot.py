import matplotlib.pyplot as plt  
import matplotlib.finance as mpf 
# import seaborn as sns


def plot_candle(quotes):
    # 创建一个子图   
    fig, ax = plt.subplots(facecolor=(0.5, 0.5, 0.5))  
    fig.subplots_adjust(bottom=0.2)  
    # 设置X轴刻度为日期时间  
    # ax.xaxis_date()
    ax.autoscale_view() 
    # X轴刻度文字倾斜45度  
    plt.xticks(rotation=45)  
    plt.title(u"601558")  
    plt.xlabel(u"time")  
    plt.ylabel(u"price")  
    mpf.candlestick_ohlc(ax,quotes,width=240,colorup='r',colordown='green')  
    plt.grid(True)
    plt.show()