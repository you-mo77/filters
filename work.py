import numpy as np
import librosa as lib
import soundfile as sf
import pyaudio as pa
import time 
from scipy import signal
import threading as th
import PySimpleGUI as sg
from os import path
from pydub import AudioSegment as AS
from pydub.utils import mediainfo
import tempfile as tf
import os
import ffmpeg

num = 1

target_fs = 48000

# 試聴用音声生成(sin波440[hz]) return wave
def make_test_sound(freq):
    length = 1
    fs = 48000
    t = np.arange(0, length, 1/fs)
    wave = np.sin(2 * np.pi * freq * t)
    wave = wave.astype(np.float32)
    return wave
 
# 音声ファイルパス
path = "crystalized_2.wav"

# 音声取得
# data, fs = lib.load(path,mono=False, sr=48000)
data = 0
fs = 48000

#data, fs = lib.load("crystalized_.5.wav", mono=False, sr=48000)

# 初期化
first1 = True
first2 = True
first3 = True
last1 = True
last2 = True
last3 = True
buffer_size = target_fs
ex_buffer1 = np.zeros((2,buffer_size))
ex_buffer2 = np.zeros((2,buffer_size))
ex_buffer3 = np.zeros((2,buffer_size))
for_filter_array1 = np.zeros((2,buffer_size * 2))
for_filter_array2 = np.zeros((2,buffer_size * 2))
for_filter_array3 = np.zeros((2,buffer_size * 2))
play_data1 = np.zeros((2,buffer_size))
play_data2 = np.zeros((2,buffer_size))
play_data3 = np.zeros((2,buffer_size))
wave1 = make_test_sound(440)
wave2 = make_test_sound(880)
wave3 = make_test_sound(7000)

# 書き込み位置
i1 = 0
i2 = 0
i3 = 0

# カットオフ周波数forスペクトル(最大周波数はサンプル周波数/2)
fc = [0,700,7000,target_fs/2 - 1]

# 出力配列(これにフィルタ後のデータをたしてく)
total1 = np.zeros((2,buffer_size))
total2 = np.zeros((2,buffer_size))
total3 = np.zeros((2,buffer_size))

# スタート位置
start_pos = 0

# 新スタート位置
s1 = 0
s2 = 0
s3 = 0 

# デバイスインデックス
l_dev = 0
m_dev = 0
h_dev = 0

# デバイス取得(hostapi's index == 2(WASAPI))
dev_index = []
dev_name = []
dev_channel = []

# test
p = pa.PyAudio()

# デバイス取得 → dev_index, dev_name　順番にappned
def get_list():
    global dev_index, dev_name, dev_channel

    # 出力デバイス分ループ
    for i in range(p.get_device_count()):

        # デバイス情報取得
        info = p.get_device_info_by_index(i)
        
        # wasapiの情報のみを取得 他のAPIインデックスは「p.get_host_api_info_by_index(i)」で取得可能 デバッグ用に別の条件式もつけている hostApiはダックをつけないと2にならない？
        if int(info["hostApi"]) == 2 and int(info["maxOutputChannels"]) == 2:
        #if int(info["index"]) < 5:
            dev_index.append(info["index"])
            dev_name.append(info["name"])
            dev_channel.append(info["maxOutputChannels"])

    # デバイスリスト制作
    dev_list = [[c1, c2, c3] for c1, c2, c3 in zip(dev_index, dev_name, dev_channel)]

    return dev_list

# gui表示用(l_index, m_index, h_indexを決めてもらう)
def gui():
    global l_dev,m_dev,h_dev,p

    # デバイスリスト取得
    dev_list = get_list()
    header = ["デバイス番号", "デバイス名", "出力チャンネル数"]
    widths = [10,30,13]

    # レイアウト
    layout = [[sg.Text("[デバイス名 : atom mini] のデバイス番号を割り当ててください", font=("Arial",20), text_color="black", background_color="white")],
              [sg.Text("低域デバイス", font=("Arial",15), text_color="black", background_color="white"), sg.Combo(values=dev_index, key="l_dev", default_value="選択してください", size=(30,1), font=("Arial",15)), sg.Button("チェック", key="低域試聴"), sg.Text("", font=("Arial",15), text_color="red", background_color="white", key="error1")],
              [sg.Text("中域デバイス", font=("Arial",15), text_color="black", background_color="white"), sg.Combo(values=dev_index, key="m_dev", default_value="選択してください", size=(30,1), font=("Arial",15)), sg.Button("チェック", key="中域試聴"), sg.Text("", font=("Arial",15), text_color="red", background_color="white", key="error2")],
              [sg.Text("高域デバイス", font=("Arial",15), text_color="black", background_color="white"), sg.Combo(values=dev_index, key="h_dev", default_value="選択してください", size=(30,1), font=("Arial",15)), sg.Button("チェック", key="高域試聴"), sg.Text("", font=("Arial",15), text_color="red", background_color="white", key="error3")],
              [sg.Table(values=dev_list, headings=header, col_widths=widths, auto_size_columns=False,font=("Arial",15), text_color="black", background_color="white")],
              [sg.Button("決定",font=("Arial",15))]]
    
    # ウィンドウ作成
    window = sg.Window('デバイス割り当て', layout, background_color="white")

    # イベントループ(デバイス指定のみ 表示し続けたほうがいい？)
    while True:
        # イベント待ち
        event, values = window.read()

        # デバイス決定
        if event == "決定":
            # デバイス代入
            if values["l_dev"] in dev_index:
                window["error1"].update("")
                l_dev = int(values["l_dev"])
            else:
                window["error1"].update("正確に指定してください")
            if values["m_dev"] in dev_index:
                window["error2"].update("")
                m_dev = int(values["m_dev"])
            else:
                window["error2"].update("正確に指定してください")
            if values["h_dev"] in dev_index:
                window["error3"].update("")
                h_dev = int(values["h_dev"])
            else:
                window["error3"].update("正確に指定してください")

            # デバイス設定終了
            if values["l_dev"] in dev_index and values["m_dev"] in dev_index and values["h_dev"] in dev_index:
                window.close()
                break
        """
        # 試聴用
        if event == "低域試聴":
            low_test(int(values["l_dev"]))
        if event == "中域試聴":
            middle_test(int(values["m_dev"]))
        if event == "高域試聴":
            high_test(int(values["h_dev"]))
        """
            
        # ウィンドウ閉じる
        if event == sg.WINDOW_CLOSED:
            break

        # 
        
    return

# 試聴用コールバック
def test_callback1(frame_count):
    wave = make_test_sound(440)
    wave = np.repeat(wave,2)
    wave = wave.tobytes("C")
    return (wave, pa.paContinue)
def test_callback2(frame_count):
    wave = make_test_sound(700)
    return (wave, pa.paContinue)
def test_callback3(frame_count):
    wave = make_test_sound(7000)
    return (wave, pa.paContinue)


# 試聴
def low_test(dev_index):
    global p

    print(dev_index)
    test_stream = p.open(format=pa.paFloat32,
                    channels=2,
                    rate=48000,
                    output_device_index=dev_index,
                    output=True,
                    stream_callback=lambda a1,b1,c1,d1:test_callback1(b1),
                    frames_per_buffer=48000)
    while test_stream.is_active():
        time.sleep(0.1)

    test_stream.close()

    return
def middle_test(dev_index):
    global p

    print(dev_index)
    test_stream = p.open(format=pa.paFloat32,
                    channels=2,
                    rate=48000,
                    output_device_index=dev_index,
                    output=True,
                    
                    stream_callback=lambda a1,b1,c1,d1:test_callback2(b1),
                    frames_per_buffer=48000)
    while test_stream.is_active():
        time.sleep(0.1)

    test_stream.close()

    return
def high_test(dev_index):
    global p

    print(dev_index)
    test_stream = p.open(format=pa.paFloat32,
                    channels=2,
                    rate=48000,
                    output_device_index=dev_index,
                    output=True,
                    stream_callback=lambda a1,b1,c1,d1:test_callback3(b1),
                    frames_per_buffer=48000)
    while test_stream.is_active():
        time.sleep(0.1)

    test_stream.close()

    return

# フォーマット変更(-> .wav)
def to_wav(path:str):

    if path.lower().endswith(".wav"):
        print("this is wav")
        data, fs = lib.load(path, mono=False, sr=None)

    else:
        print("this is not wav")
        info = mediainfo(path)
        fs = info["sample_rate"]
        with tf.TemporaryDirectory() as dirname:
            print(dirname)
            t_stream = ffmpeg.input(path)
            t_stream = ffmpeg.output(t_stream, f"{dirname}/test.wav")
            ffmpeg.run(t_stream)
            data, fs = lib.load(f"{dirname}/test.wav",mono=False,sr=None)

    return data, fs

####test####
#gui()
#print(f"l_dev:{l_dev}")
#print(f"m_dev:{m_dev}")
#print(f"h_dev:{h_dev}")
#exit()
####****####

"""
#再生関数(各々でストリームを開く)
def play1():
    p1 = pa.PyAudio()
    stream1 = p1.open(format=pa.paFloat32,
                    channels=2,
                    rate=fs,
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
                    rate=fs,
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
                    rate=fs,
                    output=True,
                    stream_callback=lambda a3,b3,c3,d3:callback3(b3),
                    frames_per_buffer=buffer_size)
    
    while stream3.is_active():
        time.sleep(0.1)

    stream3.close()
    p3.terminate()
    
    return
"""

# 新コールバック関数(各々の中でフィルタしてそれを返す )
def callback1(frame_count):
    global s1,for_filter_array1,ex_buffer1,first1,total1,play_data1,i1

    # 切り出し
    output = data[:,s1:s1+fs]

    # リサンプリング
    output = lib.resample(y=output, orig_sr=fs, target_sr=target_fs)

    # フィルタリングデータ代入配列(前回バッファと今回バッファ結合->クロスフェード用に山なりに変化させる)
    for_filter_array1 = np.hstack((ex_buffer1,output))
    if int(output.shape[1]) != 0:
        for i in range(buffer_size):
            for_filter_array1[:,i] *= (i + 1)/buffer_size
            #for_filter_array[:,int(for_filter_array.shape[1]) - 1 - i] *= ((i + 1)/buffer_size)
        for i in range(int(output.shape[1])):
            for_filter_array1[:,buffer_size + i] *= (buffer_size - i)/buffer_size
    else:
        for i in range(int(ex_buffer1.shape[1])):
            for_filter_array1[:,i] *= (i + 1)/buffer_size
    
    # ex_bufferに今回のデータを記憶
    ex_buffer1 = output

    # フィルタリング
    filtered_data1 = np.zeros(for_filter_array1.shape)

    # フィルタリング(データが存在する場合)
    if for_filter_array1.shape[1] != 0:
        #フィルタリング
        filtered_data1[0] = filter1(fc,for_filter_array1[0])
        filtered_data1[1] = filter1(fc,for_filter_array1[1])

        # dtype修正
        #print(f"filtered_data1:{filtered_data.dtype}")
        filtered_data1 = filtered_data1.astype(np.float32)

        # 正規化チックなもの(クリッピング対策)もとの音声よりかなりボリュームが減っているため、出力側で問題がなければ消すべき
        filtered_data1 /= 2

        # 出力ファイルに追加(事前にバッファ分確保 1回のコールバックごとにバッファ分確保して、1バッファ分前から加算していく 最終的には消していい デバッグ用)
        if first1:
            total1[:,s1:(s1 + int(for_filter_array1.shape[1]))] += filtered_data1[:,buffer_size:]
            first1 = False
        else:
            total1 = np.append(total1,np.zeros(output.shape), axis=1)
            total1[:,i1 - buffer_size:(i1 + int(for_filter_array1.shape[1]))] += filtered_data1
            play_data1 = total1[:,i1 - buffer_size : i1].astype(np.float32)

        # 転置(行と列入れ替え[左1, 右1],[左2, 右2],・・・)
        play_data1 = play_data1.T

        # １行に([左1, 右1, 左2, 右2, ・・・])
        play_data1 = np.ravel(play_data1)

        # byte形式に変換
        byte_data = play_data1.tobytes("C")

        # 切り出し位置更新
        s1 += fs

        # 書き込み位置更新
        i1 += target_fs

        # callback終了
        return (byte_data, pa.paContinue)
    else:
        return (b'',pa.paContinue)  

def callback2(frame_count):
    global s2,for_filter_array2,ex_buffer2,first2,total2,play_data2,i2

    # 切り出し
    output = data[:,s2:s2+fs]

    # リサンプリング
    output = lib.resample(y=output, orig_sr=fs, target_sr=target_fs)

    # フィルタリングデータ代入配列(前回バッファと今回バッファ結合->クロスフェード用に山なりに変化させる)
    for_filter_array2 = np.hstack((ex_buffer2,output))
    if int(output.shape[1]) != 0:
        for i in range(buffer_size):
            for_filter_array2[:,i] *= (i + 1)/buffer_size
            #for_filter_array[:,int(for_filter_array.shape[1]) - 1 - i] *= ((i + 1)/buffer_size)
        for i in range(int(output.shape[1])):
            for_filter_array2[:,buffer_size + i] *= (buffer_size - i)/buffer_size
    else:
        for i in range(int(ex_buffer2.shape[1])):
            for_filter_array2[:,i] *= (i + 1)/buffer_size
    
    # ex_bufferに今回のデータを記憶
    ex_buffer2 = output

    # フィルタリング
    filtered_data2 = np.zeros(for_filter_array2.shape)

    # フィルタリング(データが存在する場合)
    if for_filter_array2.shape[1] != 0:
        # フィルタリング
        filtered_data2[0] = filter2(fc,for_filter_array2[0])
        filtered_data2[1] = filter2(fc,for_filter_array2[1])

        # dtype修正
        #print(f"filtered_data1:{filtered_data.dtype}")
        filtered_data2 = filtered_data2.astype(np.float32)

        # 正規化チックなもの(クリッピング対策)もとの音声よりかなりボリュームが減っているため、出力側で問題がなければ消すべき
        filtered_data2 /= 2

        # 出力ファイルに追加(事前にバッファ分確保 1回のコールバックごとにバッファ分確保して、1バッファ分前から加算していく 最終的には消していい デバッグ用)
        print(f"total2:{total2.shape}")
        print(f"filtered_data2:{filtered_data2.shape}")
        print(f"i2:{i2}")
        print(f"first2:{first2}")
        if first2:
            total2[:,i2:(i2 + int(for_filter_array2.shape[1]))] += filtered_data2[:,buffer_size:]
            first2 = False
        else:
            total2 = np.append(total2,np.zeros(output.shape), axis=1)
            total2[:,(i2 - buffer_size):(i2 + int(for_filter_array2.shape[1]))] += filtered_data2
            play_data2 = total2[:,i2 - buffer_size : i2].astype(np.float32)

        # 転置(行と列入れ替え[左1, 右1],[左2, 右2],・・・)
        play_data2 = play_data2.T

        # １行に([左1, 右1, 左2, 右2, ・・・])
        play_data2 = np.ravel(play_data2)

        # byte形式に変換
        byte_data = play_data2.tobytes("C")

        # 切り出し位置更新
        s2 += fs

        # 書き込み位置更新
        i2 += target_fs

        # callback終了
        return (byte_data, pa.paContinue)
    else:
        return (b'',pa.paContinue)  

def callback3(frame_count):
    global s3,for_filter_array3,ex_buffer3,first3,total3,play_data3,i3

    # 切り出し
    output = data[:,s3:s3+fs]

    # リサンプリング
    output = lib.resample(y=output, orig_sr=fs, target_sr=target_fs)

    #フィルタリングデータ代入配列(前回バッファと今回バッファ結合->クロスフェード用に山なりに変化させる)
    for_filter_array3 = np.hstack((ex_buffer3,output))
    if int(output.shape[1]) != 0:
        for i in range(buffer_size):
            for_filter_array3[:,i] *= (i + 1)/buffer_size
            #for_filter_array[:,int(for_filter_array.shape[1]) - 1 - i] *= ((i + 1)/buffer_size)
        for i in range(int(output.shape[1])):
            for_filter_array3[:,buffer_size + i] *= (buffer_size - i)/buffer_size
    else:
        for i in range(int(ex_buffer3.shape[1])):
            for_filter_array3[:,i] *= (i + 1)/buffer_size
    
    #ex_bufferに今回のデータを記憶
    ex_buffer3 = output

    #フィルタリング
    filtered_data3 = np.zeros(for_filter_array3.shape)

    #フィルタリング(データが存在する場合)
    if for_filter_array3.shape[1] != 0:
        #フィルタリング
        filtered_data3[0] = filter3(fc,for_filter_array3[0])
        filtered_data3[1] = filter3(fc,for_filter_array3[1])

        #dtype修正
        filtered_data3 = filtered_data3.astype(np.float32)

        #正規化チックなもの(クリッピング対策)もとの音声よりかなりボリュームが減っているため、出力側で問題がなければ消すべき
        filtered_data3 /= 2

        #出力ファイルに追加(事前にバッファ分確保 1回のコールバックごとにバッファ分確保して、1バッファ分前から加算していく 最終的には消していい デバッグ用)
        if first3:
            total3[:,i3:(i3 + int(for_filter_array3.shape[1]))] += filtered_data3[:,buffer_size:]
            first3 = False
        else:
            total3 = np.append(total3,np.zeros(output.shape), axis=1)
            total3[:,i3 - buffer_size:(i3 + int(for_filter_array3.shape[1]))] += filtered_data3
            play_data3 = total3[:,i3 - buffer_size : i3].astype(np.float32)

        # 転置(行と列入れ替え[左1, 右1],[左2, 右2],・・・)
        play_data3 = play_data3.T

        # １行に([左1, 右1, 左2, 右2, ・・・])
        play_data3 = np.ravel(play_data3)

        # byte形式に変換
        byte_data = play_data3.tobytes("C")

        #切り出し位置更新
        s3 += fs

        # 書き込み位置更新
        i3 += target_fs

        #callback終了
        return (byte_data, pa.paContinue)
    else:
        return (b'',pa.paContinue)  

"""
#コールバック関数
def callback(frame_count,fc):
    global total1,total2,total3
    global start_pos, data, fs
    global ex_buffer 
    global first_callback
    
    # 切り出す(ステレオ->2行frame_count列)
    output_data = data[:, start_pos:(start_pos + frame_count)]

    #フィルタリングデータ代入配列(前回バッファと今回バッファ結合->クロスフェード用に山なりに変化させる)
    for_filter_array = np.hstack((ex_buffer,output_data))
    if int(output_data.shape[1]) != 0:
        for i in range(buffer_size):
            for_filter_array[:,i] *= (i + 1)/buffer_size
            #for_filter_array[:,int(for_filter_array.shape[1]) - 1 - i] *= ((i + 1)/buffer_size)
        for i in range(int(output_data.shape[1])):
            for_filter_array[:,buffer_size + i] *= (buffer_size - i)/buffer_size
    else:
        for i in range(int(ex_buffer.shape[1])):
            for_filter_array[:,i] *= (i + 1)/buffer_size
    
    #ex_bufferに今回のデータを記憶
    ex_buffer = output_data

    #フィルタリング
    filtered_data1 = np.zeros(for_filter_array.shape)
    filtered_data2 = np.zeros(for_filter_array.shape)
    filtered_data3 = np.zeros(for_filter_array.shape)

    #再生用データ(無音)
    zero = np.zeros(2*frame_count,dtype=np.float32).tobytes('C')

    #フィルタリング(データが存在する場合)
    if for_filter_array.shape[1] != 0:
        #フィルタリング
        filtered_data1[0] = filter1(fc,for_filter_array[0])
        filtered_data1[1] = filter1(fc,for_filter_array[1])
        filtered_data2[0] = filter2(fc,for_filter_array[0])
        filtered_data2[1] = filter2(fc,for_filter_array[1])        
        filtered_data3[0] = filter3(fc,for_filter_array[0])
        filtered_data3[1] = filter3(fc,for_filter_array[1])

        #dtype修正
        #print(f"filtered_data1:{filtered_data.dtype}")
        filtered_data1 = filtered_data1.astype(np.float32)
        filtered_data2 = filtered_data2.astype(np.float32)
        filtered_data3 = filtered_data3.astype(np.float32)

        #正規化チックなもの(クリッピング対策)もとの音声よりかなりボリュームが減っているため、出力側で問題がなければ消すべき
        filtered_data1 /= 2
        filtered_data2 /= 2
        filtered_data3 /= 2

        #出力ファイルに追加(事前にバッファ分確保 1回のコールバックごとにバッファ分確保して、1バッファ分前から加算していく)
        if first_callback:
            total1[:,start_pos:(start_pos + int(for_filter_array.shape[1]))] += filtered_data1[:,buffer_size:]
            total2[:,start_pos:(start_pos + int(for_filter_array.shape[1]))] += filtered_data2[:,buffer_size:]
            total3[:,start_pos:(start_pos + int(for_filter_array.shape[1]))] += filtered_data3[:,buffer_size:]
            first_callback = False
        else:
            total1 = np.append(total1,np.zeros(output_data.shape), axis=1)
            total1[:,start_pos - buffer_size:(start_pos + int(for_filter_array.shape[1]))] += filtered_data1
            total2 = np.append(total2,np.zeros(output_data.shape), axis=1)
            total2[:,start_pos - buffer_size:(start_pos + int(for_filter_array.shape[1]))] += filtered_data2
            total3 = np.append(total3,np.zeros(output_data.shape), axis=1)
            total3[:,start_pos - buffer_size:(start_pos + int(for_filter_array.shape[1]))] += filtered_data3

        # 転置(行と列入れ替え[左1, 右1],[左2, 右2],・・・)
        filtered_data1 = filtered_data1.T

        #byte形式に直すのがうまくいってない(フィルタ後の音声はきれいに聞こえた) 
        # １行に([左1, 右1, 左2, 右2, ・・・])
        filtered_data1 = np.ravel(filtered_data1)

        #切り出し位置更新UnboundLocalError: cannot access local variable 'first_callback' where it is not associated with a value
        start_pos += frame_count

        #callback終了
        return (zero, pa.paContinue)
    else:
        return (b'',pa.paContinue)        
"""
        
# 再生関数3つ
def play1():
    global p
    #p1 = pa.PyAudio()
    stream1 = p.open(format=pa.paFloat32,
                    channels=2,
                    rate=target_fs,
                    output=True,
                    output_device_index=l_dev,
                    stream_callback=lambda a1,b1,c1,d1:callback1(b1),
                    frames_per_buffer=buffer_size)
    
    while stream1.is_active():
        time.sleep(0.1)

    stream1.close()
    #p1.terminate()

    return

def play2():
    global p
    #p2 = pa.PyAudio()
    stream2 = p.open(format=pa.paFloat32,
                    channels=2,
                    rate=target_fs,
                    output=True,
                    output_device_index=m_dev,
                    stream_callback=lambda a2,b2,c2,d2:callback2(b2),
                    frames_per_buffer=buffer_size)
    
    while stream2.is_active():
        time.sleep(0.1)

    stream2.close()
    #nate()

    return

def play3():
    global p
    #p3 = pa.PyAudio()
    stream3 = p.open(format=pa.paFloat32,
                    channels=2,
                    rate=target_fs,
                    output=True,
                    output_device_index=h_dev,
                    stream_callback=lambda a3,b3,c3,d3:callback3(b3),
                    frames_per_buffer=buffer_size)
    
    while stream3.is_active():
        time.sleep(0.1)

    stream3.close()
    #p3.terminate()
    
    return


#新フィルタ関数
def filter1(fc, data:np.ndarray):
    
    nyquist_freq = 0.5 * target_fs
    normal_cutoff = fc[1] / nyquist_freq
    b, a = signal.butter(7, normal_cutoff, btype='low')
    filtered_signal = signal.filtfilt(b, a, data)

    return filtered_signal
def filter2(fc, data:np.ndarray):
    
    nyquist_freq = 0.5 * target_fs
    normal_cutoff = fc[2] / nyquist_freq
    b, a = signal.butter(7, normal_cutoff, btype='low')
    filtered_signal = signal.filtfilt(b, a, data)
    normal_cutoff = fc[1] / nyquist_freq
    b, a = signal.butter(7, normal_cutoff, btype='high')
    filtered_signal = signal.filtfilt(b, a, filtered_signal)

    return filtered_signal
def filter3(fc, data:np.ndarray):
    nyquist_freq = 0.5 * target_fs
    normal_cutoff = fc[2] / nyquist_freq
    b, a = signal.butter(7, normal_cutoff, btype='high')
    filtered_signal = signal.filtfilt(b, a, data)

    return filtered_signal

#main関数
def main():
    global fc, fs, data

    #pyaudioインスタンス
    p = pa.PyAudio()

    # guiよりポート等々取得
    gui()

    # データ取得と変換
    data, fs = to_wav(path)

    #カットオフ周波数forスペクトル(最大周波数はサンプル周波数/2)
    fc = [0,700,7000,target_fs/2 - 1]


    #ストリーム(非同期処理),
    t1 = th.Thread(target=play1)
    t2 = th.Thread(target=play2)
    t3 = th.Thread(target=play3)
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()

    p.terminate()
    
    #音声ファイル出力(テスト用　実際はいらない)
    sf.write(f"{num}_output1.new.wav",total1.T,fs,format="wav")
    sf.write(f"{num}_output2.new.wav",total2.T,target_fs,format="wav")
    sf.write(f"{num}_output3.new.wav",total3.T,fs,format="wav")

    """
    #終了処理
    stream.close()
    p.terminate()
    """

    return 

#main
if __name__ == "__main__":
    main()