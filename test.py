import librosa as lib
import numpy as np
import soundfile as sf
from scipy import signal

"""
#a = np.arange(6).reshape(2,3)
#print(a)
#[[0 1 2]
# [3 4 5]]

#b = np.append(a,a,axis = 1)
#print(b)

#入出力はおｋ
data, fs = lib.load("crystalized.short.wav",sr = 48000,mono=False)

#カットオフ周波数
fc = 7000

#フィルタ定義
order = 7
fn = fs * 0.5
low = fc / fn
b, a = signal.butter(order, low, "low")

#フィルタかける
output_data = signal.filtfilt(b, a, data)

sf.write(f"output_order={order}.wav",output_data.T,fs)
"""

a = np.array([0,1,2,3,4])
print(f"a:{a}")
b = a[1:5]
print(f"b:{b}")