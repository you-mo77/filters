import numpy as np
from scipy import signal
import librosa as lib
import soundfile as sf

data, fs = lib.load("input.true.wav", sr = 48000)
fc = 700

# バターワースローパスフィルタを設計
nyquist_freq = 0.5 * fs
normal_cutoff = fc / nyquist_freq
b, a = signal.butter(5, normal_cutoff, btype='low')

t = np.linspace(0, 1, fs, False)
filtered_signal = signal.lfilter(b, a, data)
