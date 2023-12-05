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

"""
#a = np.arange(6).reshape(2,3)
#print(a)
#[[0 1 2]
# [3 4 5]]


#フィルタ定義
order = 7
fn = fs * 0.5
low = fc / fn
b, a = signal.butter(order, low, "low")

#フィルタかける
output_data = signal.filtfilt(b, a, data)

sf.write(f"output_order={order}.wav",output_data.T,fs)
"""
"""
a = np.array([0,1,2,3,4])
print(f"a:{a}")
b = a[1:]
print(f"b:{b}")
"""
"""
data1, fs1 = lib.load("3_output1.new.wav", sr = 48000, mono = False)
data2, fs2 = lib.load("3_output2.new.wav", sr = 48000, mono = False)
data3, fs3 = lib.load("3_output3.new.wav", sr = 48000, mono = False)
"""
"""
text1 = "this is the text1"
text2 = "this is the text2"
text3 = "this is the text3"

a = np.array([0,1,2,3])

def delay(c_num:int):
    num = 0
    for i in range(0,1000000000):
        num = i % 1000000
        if num == 0:
            print(f"Here is t{c_num}")

    return

def task(c_num,a:np.ndarray):
    if c_num == 1:
        print(f"num1:{a}")
    elif c_num == 2:
        print(f"num2:{a}")
    elif c_num == 3:
        print(f"num3:{a}")
    return 

t1 = th.Thread(target=task, args=(1,a,))
t2 = th.Thread(target=task, args=(2,a,))
t3 = th.Thread(target=task, args=(3,a,))

def main():
    #マルチスレッド
    a = np.array([0,1,2,3])data, fs = lib.load("1_output1.new.wav",sr = 48000, mono = False)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

    print("finish")

    return 
"""
"""
data1, fs1 = lib.load("crystalized_1.wav", sr=48000, mono=False)
data2, fs2 = lib.load("crystalized_3.wav", sr=48000, mono=False)
data3, fs3 = lib.load("crystalized_6.wav", sr=48000, mono=False)

s1 = 0
s2 = 0
s3 = 0

buffer_size = 8192

"""
def main():
    """
    p1 = pa.PyAudio()
    stream1 = p1.open(format=pa.paFloat32,
                    channels=2,
                    rate=fs1,
                    output=True,
                    stream_callback=lambda a1,b1,c1,d1:callback1(b1),
                    frames_per_buffer=buffer_size)
    p2 = pa.PyAudio()
    stream2 = p2.open(format=pa.paFloat32,
                    channels=2,
                    rate=fs2,
                    output=True,
                    stream_callback=lambda a2,b2,c2,d2:callback2(b2),
                    frames_per_buffer=buffer_size)
    p3 = pa.PyAudio()
    stream3 = p3.open(format=pa.paFloat32,
                    channels=2,
                    rate=fs3,
                    output=True,
                    stream_callback=lambda a3,b3,c3,d3:callback3(b3),
                    frames_per_buffer=buffer_size)
    """

    """
    t1 = th.Thread(target=play1)
    t2 = th.Thread(target=play2)
    t3 = th.Thread(target=play3)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
    """

""" 
def play1():
    p1 = pa.PyAudio()
    stream1 = p1.open(format=pa.paFloat32,
                    channels=2,
                    rate=fs1,
                    output=True,
                    stream_callback=lambda a1,b1,c1,d1:callback1(b1),
                    frames_per_buffer=buffer_size)
    
    while stream1.is_active():
        time.sleep(0.1)

    stream1.close()
    p1.terminate()

    return


def play2():
    p2 = pa.PyAudio()
    stream2 = p2.open(format=pa.paFloat32,
                    channels=2,
                    rate=fs2,
                    output=True,
                    stream_callback=lambda a2,b2,c2,d2:callback2(b2),
                    frames_per_buffer=buffer_size)
    
    while stream2.is_active():
        time.sleep(0.1)

    stream2.close()
    p2.terminate()

    return
def play3():
    p3 = pa.PyAudio()
    stream3 = p3.open(format=pa.paFloat32,
                    channels=2,
                    rate=fs3,
                    output=True,
                    stream_callback=lambda a3,b3,c3,d3:callback3(b3),
                    frames_per_buffer=buffer_size)
    
    while stream3.is_active():
        time.sleep(0.1)

    stream3.close()
    p3.terminate()
    
    return

def callback1(frame_count):
    global s1
    output = data1[:,s1:s1+frame_count]
    s1 += frame_count
    output = output.T
    output = np.ravel(output).astype(np.float32)
    byte_data = output.tobytes("C")
    print(f"1:{s1}")
    return(byte_data,pa.paContinue)

def callback2(frame_count):
    global s2
    output = data2[:,s2:s2+frame_count]
    s2 += frame_count
    output = output.T
    output = np.ravel(output).astype(np.float32)
    byte_data = output.tobytes("C")
    print(f"2:{s2}")
    return(byte_data,pa.paContinue)

def callback3(frame_count):
    global s3
    output = data3[:,s3:s3+frame_count]
    s3 += frame_count
    output = output.T
    output = np.ravel(output).astype(np.float32)
    byte_data = output.tobytes("C")
    print(f"3:{s3}")
    return(byte_data,pa.paContinue)

if __name__ == '__main__':
    main()


#bytedata = data.tobytes("C")
"""
"""
a = np.zeros(0)
a = np.append(a,"ok?")
a = np.append(a, "いいね！")
print(a)
"""

a = [1,2,3]
b = [4,5,6]
c = [7,8,9]

abc1 = [a,b,c]
abc2 = [[col1, col2, col3] for col1, col2, col3 in zip(a,b,c)]
print(abc1)
print(abc2)