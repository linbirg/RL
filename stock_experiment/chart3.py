from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import DateFormatter
from simulator import Simulator


class Chart3(Simulator):
    def __init__(self, data, maxt=150):
        super(Chart3, self).__init__(data)

        self.queue = None
        self.acct = 0
        self.positions = None
        self.txt = None

        self.maxt = maxt
        # self.data = data
        self.result = []

        # Parse the data columns
        self.tdata = [r[0] for r in self.result]
        self.ldata = [r[4] for r in self.result]
        self.hdata = [r[3] for r in self.result]
        

        # Initialize plot frame
        xfmt = DateFormatter('%Y-%m-%d %H:%M')
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
        # self.clear_lines_and_patchs()
        rect.set_alpha(1)
        self.ax.add_line(vline)
        self.ax.add_patch(rect)
        self.ax.autoscale_view()
        print(len(self.ax.lines))

    def clear_lines_and_patchs(self):
        if len(self.ax.lines) > 2*self.maxt:
            for i in range(self.maxt):
                self.ax.lines.pop(0)
                self.ax.patchs.pop(0)

    def plot(self):
        if len(self.tdata) > self.maxt:  # roll the arrays
            self.tdata = self.tdata[-self.maxt:]
            self.ldata = self.ldata[-self.maxt:]
            self.hdata = self.hdata[-self.maxt:]
            self.result = self.result[-self.maxt:]

        # Plot the next set of line data
        # line_min = Line2D(self.tdata, self.ldata, color='r')
        # self.ax.add_line(line_min)
        # line_max = Line2D(self.tdata, self.hdata, color='g')
        # self.ax.add_line(line_max)

        # Plot the next set of candlestick data
        # candlestick(self.ax, self.result, width=180,  colorup='r', colordown='g')
        self.plot_candle(self.latest)

        # Update the x-axis time date and limits
        if len(self.tdata) > 1:
            self.ax.set_xlim(self.tdata[0], self.tdata[-1]+(self.tdata[-1]-self.tdata[0])/10)

        # Update the y-axis limits
        self.ax.set_ylim(min(self.ldata) * 0.999, max(self.hdata) * 1.001)

        self.plot_position()
        # plt.draw()
        # plt.show(block=False)

    def plot_position(self):
        s = "acct:{0} positions:{1}".format(self.acct,len(self.positions))
        plt.legend([s])
        # if self.txt is None:
        #     self.txt = plt.text(self.tdata[0],max(self.hdata),s)
        # else:
        #     self.txt.set_position(self.tdata[0], max(self.hdata))
        #     self.txt.set_text(s)
        # plt.draw()

    def update_prices(self, cnt):
        """
        adds a data point from data to result list for plotting
        @return:
        """
        # print(cnt)
        # results = self.data[self.maxt + cnt]  # add another point of data
        # self.next()
        self.get_next()
        # print("step ",next_step)
        t = (self.latest[0])
        # print("step ",self.step)
        if len(self.tdata)==0 or self.tdata[-1] != t:
            trade = self.latest
            self.result.append([t, trade[1], trade[2], trade[3], trade[4]])
            self.tdata.append(self.result[-1][0]) # open
            self.ldata.append(self.result[-1][3]) # low
            self.hdata.append(self.result[-1][2]) # high

            self.plot()

    def update_plot(self, i):
        try:
            self.update_prices(cnt=i)
            # self.plot()
        except:
            pass
    
    def get_next(self):
        if self.queue is None:
            return None
        
        st_ = self.queue.get()
        print("st_",st_)
        self.set_step(st_["step"])
        self.acct = st_["acct"]
        self.positions = st_["positions"]

        return st_

    def animate(self,queue):
        if self.queue is None:
            self.queue = queue
        # With try ... except in update_plot()
        # anim = animation.FuncAnimation(fig=self.fig, func=self.update_plot, interval=1000)

        # Without try ... except in update_plot()
        anim = animation.FuncAnimation(fig=self.fig, func=self.update_plot, interval=200,
                                       frames=len(self.data)-self.maxt, repeat=False)
        plt.show()
        # plt.pause(0.5)