import numpy as np
import matplotlib.pylab as plt
from scipy.signal import hilbert, chirp

duration = 2.0
fs = 400.0
dt = 1.0 / fs


samples = int(fs*duration)
t = np.arange(samples) / fs
signal = chirp(t, 20.0, t[-1], 100.0)
signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t))

analytic_signal = hilbert(signal)

print(analytic_signal)