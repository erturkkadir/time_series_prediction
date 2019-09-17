from sim_lstm import MyLSTM
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

np.random.seed(1337)        # for reproducibility

# Definitions ##########################
# delta_t = 5                 # sampling rate in minute ie every 5 min
# train_size = 256            # Training data set length 128 points * 5 min = 10.6 h
# time_step = 5               # 6 * 5 = 0.5 Hours, window transition duration
# output_size = 4             # points in minute ie next 5 min * 4 points = 20 min
###############################################
instruments = ['EUR_USD',  'EUR_GBP', 'NZD_USD', 'USD_CAD', 'EUR_CAD']
delta_ts = [5, 10, 15, 30]          # 1 min 5 min etc
train_sizes = [64, 128, 256]        # Train data sizes
time_steps = [16, 20, 30]           # Time steps within data
output_size = 3                     # Number of points to estimate

n1_cells = 8                        # LSTM first cell
n2_cells = 2                        # LSTM second cell

target_reward = 4                   # target reward
critic_loss = 6                     # Critic loss

step = 1                            # shift 1 steps to test each time

lstm = MyLSTM()                     # LSTM Functions            # Server to save logs
plt.ion()

for inst in instruments:            # All pairs
    for delta_t in delta_ts:        # All sampling rates
        f_name = 'data/M' + str(delta_t) + "/" + inst + '.csv'
        pd = pd.read_csv(f_name,  usecols=['time', 'open', 'close', 'high', 'low', 'volume'])
        pd_data = np.array(pd)
        data_len = len(pd_data)     # 2048
        x = np.arange(len(pd_data))
        for train_size in train_sizes:  # 64
            for time_step in time_steps:    # 3
                epochs = 20
                batch_size = train_size

                lstm.init_params(epochs=epochs, n1_cells=n1_cells, n2_cells=n2_cells, delta_t=delta_t,
                                 train_size=train_size, time_steps=time_step, output_size=output_size,
                                 batch_size=batch_size)

                res = []
                # for i in range(0, data_len/10, step):      # 0 .. 2048 step 10
                for i in range(1000, 1070, step):  # 0 .. 2048 step 10
                    idx_sta = i
                    idx_end = i + train_size
                    tmp_ = pd_data[idx_sta:idx_end, :]
                    print("TMP Shape : ", tmp_.shape)  #
                    lstm.set_data(tmp_)  # set lstm data
                    current, futures = lstm.lstm_opinion()  # What is LSTM opinion
                    futures = futures[0, :]
                    print("Current : ", current)
                    print("Futures : ", futures)
                    actual = pd_data[idx_end:idx_end + output_size, 2]
                    print("Actual : ", actual)

                    est_pip = 10000 * (current - futures[-1])
                    act_pip = 10000 * (current - actual[-1])
                    error_term = est_pip - act_pip

                    sum_pip = error_term

                    print("Estimated PIP Changes : ", est_pip)
                    print("Actual    PIP Changes : ", act_pip)
                    print("Error in Estimation   : ", error_term)
                    current_time = pd_data[idx_sta:idx_end, 1]

                    # server.save_log(inst, delta_t, train_size, time_step, output_size, idx_sta, idx_end,
                    #                 est_pip, act_pip, error_term)
                    print("Inst train_size delta_t", inst, train_size, delta_t)
                    res.append([current, futures[-1]])

                # plt.plot(x, pd.close.value)
                plt.plot(res)
                plt.show(block=True)
