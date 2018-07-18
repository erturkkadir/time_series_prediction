import pandas as pd
import numpy as np
from stockstats import StockDataFrame
from pandas import Series
import matplotlib.pyplot as plt
from scipy.signal import wiener

#  https://github.com/jealous/stockstats/blob/master/README.rst
#  buy = 1, none = 0 sell = -1


class Indicators(object):

    def __init__(self):
        super(Indicators, self).__init__()
        self.stock = []
        self.data = []
        self.current = 0
        self.d_len = 0
        self.close = []

    def set_data(self, tmp):
        self.d_len = len(tmp)
        self.data = pd.DataFrame(tmp)
        self.stock = StockDataFrame.retype(self.data)
        self.close = self.stock['close']
        self.current = self.data.values[self.d_len - 1]  # Last close value

    def get_result(self):

        cur_min = 30
        cur_max = 70
        tar_min = -1
        tar_max = +1
        self.close = self.smooth()

        # rsi calculation
        rsi = self.stock['rsi_12'].fillna(0)
        # new_rsi = ((rsi - cur_min) / (cur_max - cur_min)) * (tar_max - tar_min) + tar_min
        new_rsi = [1 if item > 70 else -1 if item < 30 else 0 for item in rsi]

        # mac calculation
        mac = self.stock['macdh'].fillna(0)         # macdh is the better one
        a_sign = np.sign(mac)
        new_mac = (a_sign * ((np.roll(a_sign, 1) - a_sign) != 0).astype(int)).astype(int)

        # bollinger bands
        boll_lb = self.stock['boll_lb'].fillna(0)
        boll_ub = self.stock['boll_ub'].fillna(0)

        new_lb = np.greater(boll_lb, self.close) * -1
        new_ub = np.greater(self.close, boll_ub) * +1

        x = np.arange(self.d_len)
        fig, ax1 = plt.subplots()
        y = self.smooth()

        ax1.set_xlabel('tmp')
        ax1.set_ylabel('raw', color='blue')
        ax1.plot(x, self.close, color='blue')
        ax1.tick_params(axis='y', labelcolor='green')
        # ax1.plot(x, y, color='cyan')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        # ax2.set_ylabel('rs', color='green')  # we already handled the x-label with ax1
        # ax2.plot(x, new_lb, color='green')

        ax2.plot(x, new_rsi, color='cyan')
        ax2.plot(x, new_mac, color='red')
        ax2.plot(x, new_lb, color='magenta')
        ax2.plot(x, new_lb, color='black')

        ax2.tick_params(axis='rsi', labelcolor='green')

        # plt.axhline(30, color='yellow')
        plt.axhline(0, color='yellow')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()

        res = np.array(self.close).reshape(-1, 1)
        res = np.append(res, np.array(new_rsi).reshape(-1, 1), axis=1)
        res = np.append(res, np.array(new_mac).reshape(-1, 1), axis=1)
        res = np.append(res, np.array(new_lb).reshape(-1, 1), axis=1)
        res = np.append(res, np.array(new_ub).reshape((-1, 1)), axis=1)
        return res

    def smooth(self):
        wi = wiener(self.close, mysize=int(self.d_len/24))
        wi[0:5] = wi[7]
        wi[self.d_len-5:self.d_len] = wi[self.d_len-5]

        return wi

    def ichimoku(self, tmp_):
        dataFrame = pd.DataFrame(tmp_)
        s = dataFrame.values[len(tmp_) - 1]  # Last close value
        n1 = 9
        n2 = 26
        n3 = 52

        hhv_s_n1 = Series.rolling(s, n1).max()
        hhv_s_n2 = Series.rolling(s, n2).max()
        hhv_s_n3 = Series.rolling(s, n3).max()

        llv_s_n1 = Series.rolling(s, n1).min()
        llv_s_n2 = Series.rolling(s, n2).min()
        llv_s_n3 = Series.rolling(s, n3).min()

        conv = hhv_s_n1 + llv_s_n1 / 2
        base = hhv_s_n2 + llv_s_n2 / 2
        spana = ((conv + base) / 2).shift(n2)
        spanb = ((hhv_s_n3 + llv_s_n3) / 2).shift(n2)
        lspan = s.shift(-n2)
        return conv, base, spana, spanb, lspan