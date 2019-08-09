from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter
from simulator import Simulator

from logger import Logger


class KChart(Simulator):
    logger = None
    def __init__(self, data, maxt=150):
        super(KChart, self).__init__(data)

        if KChart.logger is None:
            KChart.logger = Logger("KChart")

        self.queue = None
        self.acct = 0
        self.positions = None
        self.txt = None

        self.maxt = maxt
        self.result = []

        # Parse the data columns
        self.times = [r[0] for r in self.result]
        self.lows = [r[4] for r in self.result]
        self.highs = [r[3] for r in self.result]
        

        # Initialize plot frame
        # xfmt = DateFormatter('%Y-%m-%d %H:%M')
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(bottom=0.2)
        # self.ax.xaxis.set_major_formatter(xfmt)
        # self.ax.xaxis_date()
        self.ax.autoscale_view()
        self.ax.grid(True)
        

    def plot_candle(self,q):
        colorup = 'r'
        colordown = 'g'
        width = 180
        OFFSET = width / 2.0

        t, open, high, low, close = q[:5]
        if close >= open:
            color = colorup
            lower = open
            height = close - open
        else:
            color = colordown
            lower = close
            height = open - close

        vline = Line2D(
            xdata=(t, t), ydata=(low, high),
            color=color,
            linewidth=0.5,
            antialiased=True,
        )

        rect = Rectangle(
            xy=(t - OFFSET, lower),
            width=width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
        
        rect.set_alpha(1)
        self.ax.add_line(vline)
        self.ax.add_patch(rect)
        self.ax.autoscale_view()

        return vline, rect

    def clear_lines_and_patchs(self):
        if len(self.ax.lines) > 2*self.maxt:
            self.logger.debug(len(self.ax.lines))
            
            for i in range(self.maxt):
                self.ax.lines.pop(0)
                self.ax.patches.pop(0)

    def update_limits(self):
        if len(self.times) > 1:
            # Update the x-axis time date and limits
            self.ax.set_xlim(self.times[0], self.times[-1]+(self.times[-1]-self.times[0])/10)
            # Update the y-axis limits
            self.ax.set_ylim(min(self.lows) * 0.999, max(self.highs) * 1.001)
        
        

    def plot(self):
        if len(self.times) > self.maxt:  # roll the arrays
            self.times = self.times[-self.maxt:]
            self.lows = self.lows[-self.maxt:]
            self.highs = self.highs[-self.maxt:]
            self.result = self.result[-self.maxt:]
        
        self.update_limits()

        line,patch = self.plot_candle(self.latest)
        legned = self.plot_position()

        return line, patch, legned

    def plot_position(self):
        s = "acct:{0} positions:{1}".format(self.acct, len(self.positions))
        lengend = plt.legend([s])
        return lengend

    def update_prices(self, cnt):
        self.get_next()
        t = (self.latest[0])
        if len(self.times) == 0 or self.times[-1] != t:
            trade = self.latest
            self.result.append([t, trade[1], trade[2], trade[3], trade[4]])
            self.times.append(self.result[-1][0]) # open
            self.lows.append(self.result[-1][3]) # low
            self.highs.append(self.result[-1][2]) # high

    def update_plot(self, i):
        self.update_prices(cnt=i)
        self.clear_lines_and_patchs()
        return self.plot()
    
    def get_next(self):
        if self.queue is None:
            return None
        
        st_ = self.queue.get()
        self.logger.debug(st_)
        
        self.set_step(st_["step"])
        self.acct = st_["total"]
        self.positions = st_["positions"]

        return st_

    def animate(self,queue):
        if self.queue is None:
            self.queue = queue
        anim = animation.FuncAnimation(fig=self.fig, func=self.update_plot, interval=200,
                                       frames=len(self.data)-self.maxt, repeat=False)
        plt.show()