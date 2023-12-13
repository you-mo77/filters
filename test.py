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

def to_wav(path:str):

    if path.lower().endswith(".wav"):
        print("this is wav")
        data, fs = lib.load(path,mono=False,sr=None)

    # データ形式→<pydub.audio_segment.AudioSegment object at 0x0000015DDC6AFFD0>　これをwavに直せる？
    elif path.lower().endswith(".mp3"):
        print("this is mp3")
        mp3_data = AS.from_mp3(path)
        info = mediainfo(path)
        fs = info["sample_rate"]
        with tf.TemporaryDirectory() as dirname:
            print(dirname)
            mp3_data.export(f"{dirname}/test_output.wav", format="wav",)
            data, fs = lib.load(f"{dirname}/test_output.wav",mono=False,sr=None)

    elif path.lower().endswith(".wma"):
        print("this is flac")
    elif path.lower().endswith(".aif") or path.lower().endswith(".aiff"):
        print("this is aiff")
    else:
        print("このフォーマットは対応していません")

    return data, fs


path1 = "crystalized.wav"
path2 = "crystalized.mp3"

data1, fs1 = to_wav(path1)
data2, fs2 = to_wav(path2)

print(f"wav_data:{data1}")
print(f"wav_fs:{fs1}")
print(f"wav_data:{data2}")
print(f"wav_fs:{fs2}")

data1 = data1.T
data2 = data2.T

print(f"data1_shape:{data1.shape}")
print(f"data2_shape:{data2.shape}")
byte_data = data1.tobytes("C")
sf.write("test_wav.wav",data1,fs1)
print("ok")
byte_data = data2.tobytes("C")
sf.write("test_mp3.wav",data2,fs2)
print("okokl")