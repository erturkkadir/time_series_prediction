import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Activation
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1337)


class TsLstm(object):

    def __init__(self):
        super(TsLstm, self).__init__()
        self.scalar = MinMaxScaler(feature_range=(0, 1))
        self.data_size = 0
        self.features = 0
        self.output_size = 0
        self.batch_size = 0
        self.time_steps = 0
        self.data = []
        self.epochs = 0
        self.train_size = 0
        self.window_size = 0
        self.raw_data = []
        self.full_data = []
        self.prediction = []
        self.close = []

    def init_params(self, epochs, train_size, time_steps, output_size, batch_size):
        self.epochs = epochs
        self.train_size = train_size
        self.time_steps = time_steps
        self.output_size = output_size
        self.batch_size = batch_size

    
    # shape of tmp is (train_size, 6)
    def set_data(self, tmp):
        self.raw_data = tmp  # save original data for potential uses
        self.close = tmp[:, 2].reshape(-1, 1)  # 2nd column "close" shape is (train_size, 1)
        self.data = np.array(self.scalar.fit_transform(self.close))  # scale close column between (0, -1)
        self.data_size = len(self.data)  # train_size
        self.features = int(self.data_size / 2)

    def lstm_opinion(self):
        train_x, train_y, pred_x = self.prepare_data()
        self.batch_size = train_x.shape[0]

        model = Sequential()
        model.add(LSTM(2, batch_size=None, input_shape=(self.time_steps, self.features)))
        model.add(Dropout(0.40))
        model.add(Dense(self.output_size))
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        model.fit(train_x, train_y, batch_size=self.batch_size, shuffle=False, verbose=0, epochs=self.epochs)
        # model.summary()

        score = model.evaluate(train_x, train_y)
        print("Success Rate : %2s: %.2f" % (model.metrics_names[1], score[1] * 100))
        p_x = model.predict(pred_x)

        trans_y = self.scalar.inverse_transform(p_x)
        last_value = self.raw_data[-1]          #
        # first_value = trans_y[0, 0]           #
        # diff = last_value - first_value
        # trans_y = diff + trans_y
        # print("Full data : ", self.full_data)
        # print("Raw data : ", self.raw_data)
        # print("Pred : ", trans_y)
        self.prediction = np.array(trans_y).reshape(-1, 1)

        return trans_y

    def prepare_data(self):
        data_x, data_y, last_x = [], [], []
        half_size = int(self.data_size / 2)
        d_len = int(half_size - self.output_size - self.time_steps) + 2
        for i in range(0, d_len):
            sub_x = []
            for k in range(0, self.time_steps):
                ix1 = i + k
                ix2 = i + k + half_size
                inp_x = self.data[ix1:ix2]  # [0:127] [1:128] ....
                # print("Train data i, k ix1 ix2 ", i, k, ix1, ix2)
                df = self.raw_data[ix1:ix2]
                # indicators.set_data(df)                                 # set data for indicators
                # rsi1, rsi2, mac1, mac2, bl_up, bl_dn = indicators.get_result()
                # fft_x = self.get_fft(np.reshape(inp_x, -1))
                # fft_x[-1] = rsi1
                # fft_x[-2] = rsi2
                # fft_x[-3] = mac1
                # fft_x[-4] = mac2
                # fft_x[-5] = bl_dn
                # fft_x[-6] = bl_up
                # sub_x.append(fft_x)
            data_x.append(sub_x)
            iy1 = i + k + half_size
            iy2 = iy1 + self.output_size
            # print("y data iy1 iy2 ", iy1, iy2)            # since last item excluded
            data_y.append(self.data[iy1:iy2])
        t_x = np.array(data_x)
        t_y = np.array(data_y)
        t_y = t_y[:, :, 0]

        i = half_size - self.time_steps + 1
        sub_x = []
        for k in range(0, self.time_steps):
            ix1 = i + k
            ix2 = i + k + half_size
            inp_x = self.data[ix1:ix2]  # [0:32] [1:33] ....
            # print("Pred X data i, k ix1 ix2 ", i, k, ix1, ix2-1)
            fft_x = self.get_fft(np.reshape(inp_x, -1))
            sub_x.append(fft_x)
        last_x.append(sub_x)
        p_x = np.array(last_x)
        # print("Pred shape :", p_x.shape)
        # print("Train X Shape :",  t_x.shape)
        # print("Train Y Shape :", t_y.shape)
        return t_x, t_y, p_x

    def get_fft(self, tmp):          # MUST, make sure the length of x is 2**n
        n = len(tmp)  # 1024
        n2 = int(n / 2)
        # fft = (1.0/n) * np.abs(np.fft.fft(signal))
        tft = np.fft.fft(tmp)
        fft = np.real(tft) + np.imag(tft)
        fft[0:2] = 0
        fft[n2:n] = np.arctan2(np.real(tft[n2:n]), np.imag(tft[n2:n]))
        # fft[n2:n] = np.real(tft[n2:n]) - np.imag(tft[n2:n])
        return fft

    def draw(self):
        y1 = self.raw_data
        x1 = np.arange(len(y1))

        y2 = self.prediction
        x2 = np.arange(len(y1), len(y1)+len(y2))
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111)
        ax.plot(x1, y1, label=1)
        ax.plot(x2, y2, label=2)

        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        plt.show()
        plt.pause(0.0001)