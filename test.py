import librosa as lib
import numpy as np

a = np.arange(6).reshape(2,3)
print(a)
#[[0 1 2]
# [3 4 5]]

b = np.append(a,a,axis = 1)
print(b)