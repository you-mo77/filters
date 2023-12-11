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

def to_wav(path:str):

    if path.lower().endswith(".wav"):
        print("this is wav")
        data, fs = lib.load(path)

    elif path.lower().endswith(".mp3"):
        print("this is mp3")
        data = AS.from_mp3(path)
        fs = 48000

    elif path.lower().endswith(".flac"):
        print("this is flac")
    elif path.lower().endswith(".mov") or path.lower().endswith(".m4a") or path.lower().endswith(".alac"):
        print("this is alac(.mov or .m4a or .alac)")
    else:
        print("このフォーマットは対応していません")

    return data, fs


path1 = "crystalized.wav"
path2 = "crystalized.mp3"

data1, fs1 = to_wav(path1)
print("ok1")
data2, fs2 = to_wav(path2)
print("ok2")