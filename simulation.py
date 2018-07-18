from sim_lstm import MyLSTM
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from indicators import Indicators

np.random.seed(1337)        # for reproducibility

# Definitions ##########################
# delta_t = 5                       # sampling rate in minute ie every 5 min
# train_size = 256                  # Training data set length 128 points * 5 min = 10.6 h
# time_step = 5                     # 6 * 5 = 0.5 Hours, window transition duration
# output_size = 4                   # points in minutes ie next 5 min * 4 points = 20 min
###############################################
instruments = ['EUR_USD',  'EUR_GBP', 'NZD_USD', 'USD_CAD', 'EUR_CAD']
delta_ts = [5, 10, 15, 30]          # 1 min 5 min etc
train_sizes = [64, 128, 256]        # Train data sizes
time_steps = [16, 20, 30]           # Time steps within data
output_size = 3                     # Number of points to estimate

step = 1                            # shift 1 steps to test each time

lstm = MyLSTM()                     # LSTM Functions
indicators = Indicators()           # Indicators

for inst in instruments:            # All pairs
    for delta_t in delta_ts:        # All sampling rates
        f_name = 'data/M' + str(delta_t) + "/" + inst + '.csv'
        pd = pd.read_csv(f_name, tupleize_cols=True, usecols=['time', 'open', 'close', 'high', 'low', 'volume'])
        pd_data = np.array(pd)
        data_len = len(pd_data)     # 2048
        x = np.arange(len(pd_data))
        for train_size in train_sizes:  # 64
            for time_step in time_steps:    # 3
                epochs = 20
                batch_size = 100

                lstm.init_params(epochs=epochs, delta_t=delta_t, train_size=train_size, time_steps=time_step,
                                 output_size=output_size, batch_size=batch_size)
                res = []
                # for i in range(0, data_len/10, step):             # 0 .. 2048 step 10 every single mins
                for i in range(1000, 1070, step):                   # slice from mid day
                    idx_sta = i                                     # start index
                    idx_end = i + train_size                        # end index
                    tmp_np = pd_data[idx_sta:idx_end, :]            # shape is (time_steps, 6) numpy format
                    tmp_pd = pd.iloc[idx_sta:idx_end, 2]            # panda format

                    indicators.set_data(tmp_pd)                     # set data for indicators
                    indicators.get_result()

                    lstm.set_data(tmp_np)  # set data for LSTM object
                    current, futures = lstm.lstm_opinion()          # What is LSTM opinion
                    futures = futures[0, :]                         # estimated values, size : output_size
                    actual = pd_data[idx_end:idx_end + output_size, 2]  # what is the actual value
                    print("Current : ", current)                    #
                    print("Futures : ", futures)                    #
                    print("Actual : ", actual)                      #

                    est_pip = 10000 * (current - futures[-1])       # estimation in terms of pip
                    act_pip = 10000 * (current - actual[-1])        # actual in terms of pip
                    error_term = est_pip - act_pip                  # The error term

                    print("Estimated PIP Changes : ", est_pip)      #
                    print("Actual    PIP Changes : ", act_pip)      #
                    print("Error in Estimation   : ", error_term)   #
                    current_time = pd_data[idx_sta:idx_end, 1]      # When all happen
                    res.append([current, futures[-1]])              # Save them to plot later on

                # plt.plot(x, pd.close.value)                       # x is just an index for x axis
                plt.plot(x, res)                                    # plot both
                plt.show(block=True)                                # Wait until user interaction
