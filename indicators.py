import pandas as pd
import numpy as np
from pandas import Series
from stockstats import StockDataFrame
from scipy.signal import wiener

#  https://github.com/jealous/stockstats/blob/master/README.rst
#  buy = 1, none = 0 sell = -1


class Indicators(object):

    def __init__(self):
        super(Indicators, self).__init__()
        self.data = []
        self.current = 0
        self.d_len = 0
        self.close = []
        self.stock = []

    def set_data(self, tmp):
        self.d_len = len(tmp)
        df = pd.DataFrame(tmp)
        df.columns = ['time', 'open', 'close', 'high', 'low', 'volume']
        self.stock = StockDataFrame.retype(df)
        self.close = df.close.values

    def get_result(self):

        # rsi calculation
        rsi = self.stock['rsi_12'].fillna(0)
        # new_rsi = ((rsi - cur_min) / (cur_max - cur_min)) * (tar_max - tar_min) + tar_min
        new_rsi = [-1 if item > 70 else +1 if item < 30 else 0 for item in rsi]
        rsi1 = new_rsi[-1]      # last 2 values
        rsi2 = new_rsi[-2]

        # mac calculation
        mac = self.stock['macdh'].fillna(0)         # macdh is the better one
        a_sign = np.sign(mac)
        new_mac = (a_sign * ((np.roll(a_sign, 1) - a_sign) != 0).astype(int)).astype(int)
        mac1 = new_mac.iloc[-1]
        mac2 = new_mac.iloc[-2]

        # bollinger bands
        boll_lb = self.stock['boll_lb'].fillna(0)
        boll_ub = self.stock['boll_ub'].fillna(0)

        new_lb = np.greater(boll_lb, self.stock.close) * -1
        new_ub = np.greater(self.stock.close, boll_ub) * +1

        bl_lb = new_lb.iloc[-1]
        bl_ub = new_ub.iloc[-1]

        return rsi1, rsi2, mac1, mac2, bl_lb, bl_ub

    def graph(self):
        x = np.arange(self.d_len)
        # fig, ax1 = plt.subplots()
        # y = self.smooth()
        #
        # ax1.set_xlabel('tmp')
        # ax1.set_ylabel('raw', color='blue')
        # ax1.plot(x, self.close, color='blue')
        # ax1.tick_params(axis='y', labelcolor='green')
        # # ax1.plot(x, y, color='cyan')
        #
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        #
        # # ax2.set_ylabel('rs', color='green')  # we already handled the x-label with ax1
        # # ax2.plot(x, new_lb, color='green')
        #
        # ax2.plot(x, new_rsi, color='cyan')
        # ax2.plot(x, new_mac, color='red')
        # ax2.plot(x, new_lb, color='magenta')
        # ax2.plot(x, new_lb, color='black')
        #
        # ax2.tick_params(axis='rsi', labelcolor='green')
        #
        # # plt.axhline(30, color='yellow')
        # plt.axhline(0, color='yellow')
        #
        # fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #
        return x