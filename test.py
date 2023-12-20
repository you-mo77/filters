import librosa as lib
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import threading as th
import streamlit as st
import io
#import sounddevice as sd
import wave
import pyaudio as pa
import threading as th
import time
from pydub import AudioSegment as AS 
from pydub.utils import mediainfo
import tempfile as tf
import os
import ffmpeg

target_fs = 192000
"""
data, fs = lib.load("crystalized_1.wav",sr=None, mono=False)
print(f"shape:{data.shape}")
print(f"fs:{fs}")

new_data = lib.resample(y=data, orig_sr=fs, target_sr=target_fs)
print(new_data.shape)

sf.write("resampled_1.wav", new_data.T, target_fs)
"""

data, fs = lib.load("crystalized.wav", sr=None, mono=False)

s = 0

def callback(frame_count):
    global s
    print(frame_count)

    output = data[:, s : s+fs]
    output.astype(np.float32)

    print(output.dtype)

    s += fs

    output = lib.resample(y=output, orig_sr=fs, target_sr=target_fs)

    print(output.shape)

    tenti = output.T
    raveled = np.ravel(tenti)
    byte_data = raveled.tobytes("C")

    return(byte_data, pa.paContinue)

p = pa.PyAudio()

l_dev = 13

buffer_size = target_fs

print(fs)

stream = p.open(format=pa.paFloat32,
                    channels=2,
                    rate=target_fs,
                    output=True,
                    output_device_index=l_dev,
                    stream_callback=lambda a1,b1,c1,d1:callback(b1),
                    frames_per_buffer=buffer_size,
                    )
    
while stream.is_active():
    time.sleep(0.1)

stream.close()