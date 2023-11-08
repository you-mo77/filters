import numpy as np
import librosa as lib
import soundfile as sf
import pyaudio as pa
import time 
from scipy import signal

#音声取得
data, fs = lib.load("crystalized_short.wav",mono=False, sr=48000)
#print(f"data:{data.dtype}")
#data = data.astype(np.float64)
#print(f"data2:{data.dtype}")


total1 = np.zeros((2,0))
total2 = np.zeros((2,0))
total3 = np.zeros((2,0))

"""
n = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
])
print(n)
print(n.shape[1])
#print(n[:, 1:3])
#print(np.ravel(n[:, 1:3].transpose()))

exit()
"""

#スタート位置
start_pos = 0

#コールバック関数
def callback(frame_count,h,fc):
    global total1,total2,total3
    global start_pos, data, fs
    
    # 切り出す(ステレオ->2行frame_count列　)
    output_data = data[:, start_pos:(start_pos + frame_count)]
    #print(f"output_data:{output_data.dtype}")

    #フィルタリング
    #filtered_data = np.zeros((2, (int(h[0].shape[0]) + int(output_data[0].shape[0]) - 1)))
    filtered_data1 = np.zeros(output_data.shape)
    filtered_data2 = np.zeros(output_data.shape)
    filtered_data3 = np.zeros(output_data.shape)

    if output_data.shape[1] != 0:
        filtered_data1[0] = filter1(fc,output_data[0])
        filtered_data1[1] = filter1(fc,output_data[1])
        filtered_data2[0] = filter2(fc,output_data[0])
        filtered_data2[1] = filter2(fc,output_data[1])        
        filtered_data3[0] = filter3(fc,output_data[0])
        filtered_data3[1] = filter3(fc,output_data[1])

    #print(f"filtered_data1:{filtered_data.dtype}")
    filtered_data1 = filtered_data1.astype(np.float32)
    filtered_data2 = filtered_data2.astype(np.float32)
    filtered_data3 = filtered_data3.astype(np.float32)
    #print(f"filtered_data2:{filtered_data.dtype}")


    total1 = np.append(total1,filtered_data1, axis=1)
    total2 = np.append(total2,filtered_data2, axis=1)
    total3 = np.append(total3,filtered_data3, axis=1)

    #sf.write("in_callback.wav",total1.T,fs,format="wav")
    #print(type(filtered_data))
    #print(f"filtered_data:{filtered_data.dtype}")

    # 転置(行と列入れ替え[左1, 右1],[左2, 右2],・・・)
    filtered_data1 = filtered_data1.T

    #byte形式に直すのがうまくいってない(フィルタ後の音声はきれいに聞こえた) 
    # １行に([左1, 右1, 左2, 右2, ・・・])
    filtered_data1 = np.ravel(filtered_data1)

    #バイト形式に変換
    byte_data = filtered_data1.tobytes('C')

    #print("before ravel: ", output_data.shape)
    #切り出し位置更新
    start_pos += frame_count

    #output_data = bytes([0xff for x in range(frame_count*2)] + [0 for x in range(frame_count*2)])
    #print({"frame_count": frame_count ,"start_pos": start_pos})
    #print("after ravel: ", output_data.shape)
    #print(len(byte_data))

    #再生
    return (byte_data, pa.paContinue)
    #return(b'',pa.paContinue)
#フィルタ関数
"""
def filtering0(h: np.ndarray, data: np.ndarray):

    #フィルタ適用
    filtered0 = np.zeros((int(h[0].shape[0]) + int(data.shape[0]) - 1))
    print(data)
    filtered0 = np.convolve(data,h[0])
    
    return filtered0
def filtering1(h: np.ndarray, data: np.ndarray, range_num: int):

    #フィルタ適用
    filtered1 = np.zeros((range_num,int(h[1].shape[0]) + int(data.shape[0]) - 1))
    filtered1 = np.convolve(data,h[1])
    
    return filtered1
def filtering2(h: np.ndarray, data: np.ndarray, range_num: int):

    #フィルタ適用
    filtered2 = np.zeros((range_num,int(h[2].shape[0]) + int(data.shape[0]) - 1))
    filtered2 = np.convolve(data,h[2])
    
    return filtered2
"""
    
#新フィルタ関数
def filter1(fc, data:np.ndarray):
    
    nyquist_freq = 0.5 * fs
    normal_cutoff = fc[1] / nyquist_freq
    b, a = signal.butter(7, normal_cutoff, btype='low')
    filtered_signal = signal.filtfilt(b, a, data)

    return filtered_signal
def filter2(fc, data:np.ndarray):
    
    nyquist_freq = 0.5 * fs
    normal_cutoff = fc[2] / nyquist_freq
    b, a = signal.butter(7, normal_cutoff, btype='low')
    filtered_signal = signal.filtfilt(b, a, data)
    normal_cutoff = fc[1] / nyquist_freq
    b, a = signal.butter(7, normal_cutoff, btype='high')
    filtered_signal = signal.filtfilt(b, a, filtered_signal)

    return filtered_signal
def filter3(fc, data:np.ndarray):
    nyquist_freq = 0.5 * fs
    normal_cutoff = fc[2] / nyquist_freq
    b, a = signal.butter(7, normal_cutoff, btype='high')
    filtered_signal = signal.filtfilt(b, a, data)

    return filtered_signal

#インパルス応答作成まで
def init(tap: int, fc: list, fs: int, range_num: int):

    #タップ数に調整
    kc = [int(x / fs * tap) for x in fc]

    #スペクトル生成
    spectrum = np.zeros((range_num,tap))
    for i in range(range_num):
        spectrum[i][kc[i]:kc[i+1]] += 1.0   
        spectrum[i][tap - kc[i+1]:tap - kc[i]] += 1.0

    #インパルス応答生成
    h = np.zeros((range_num,tap))
    for i in range(range_num):
        h[i] = np.real(np.fft.ifft(spectrum[i]))
        h[i] = np.roll(h[i],int(h[i].shape[0] / 2))

    return h

#main関数
def main():
    #初期値代入
    tap = 1024
    range_num = 3

    #元の音声取得位置

    #カットオフ周波数forスペクトル(最大周波数はサンプル周波数/2)
    fc = [0,700,7000,23999]

    #インパルス応答
    h = init(tap, fc, fs, range_num)

    #pyaudioインスタンス
    p = pa.PyAudio()

    #ストリーム

    print(f"opening {data.shape[0]} channels")
    
    stream = p.open(format=pa.paFloat32,
                    channels=2,
                    rate=fs,
                    output=True,
                    stream_callback=lambda a,b,c,d:callback(b,h ,fc),
                    frames_per_buffer=48000)

    while stream.is_active():
        time.sleep(0.1)
    
    print("terminated")
    #print(total1.shape)
    sf.write("output1.wav",total1.T,fs,format="wav")
    sf.write("output2.wav",total2.T,fs,format="wav")
    sf.write("output3.wav",total3.T,fs,format="wav")


    stream.close()

    p.terminate()

    return 

#main
if __name__ == "__main__":
    main()

"""
    #初期値代入
    tap = 1024
    fc = [0,700,7000, 11025]
    fs = 22050
    range_num = 3

    #インパルス生成
    h = init(tap, fc, fs, range_num)

    #処理開始
    with wave.open("input.true.wav", "rb") as wf:

        #コールバック関数
        def callback0(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)

            filtered0 = filtering0(h, data, range_num)

            return (filtered0, pa.paContinue)
        
        def callback1(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)

            filtered1 = filtering1(h, data, range_num)

            return (filtered1, pa.paContinue)
        
        def callback2(in_data, frame_count, time_info, status):
            data = wf.readframes(frame_count)

            filtered2 = filtering0(h, data, range_num)

            return (filtered2, pa.paContinue)

        #pyaudioインスタンス
        p = pa.PyAudio()

        #ストリーム開始
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        stream_callback=callback0)
        
        while stream.is_active():
            time.sleep(0.1)

        stream.close()

        p.terminate()
"""

###debug###
#出力
"""
for i in range(range_num):
    sf.write(f"output{i}.wav",filtered[i],fs)

x = np.arange(spectrum.shape[1])
plt.scatter(x, spectrum[0])
plt.show()
plt.scatter(x, spectrum[1])
plt.show()
plt.scatter(x, spectrum[2])
plt.show()
"""