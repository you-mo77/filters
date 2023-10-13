import numpy as np
import librosa as lib
import scipy as sci
import soundfile as sf
import matplotlib.pyplot as plt
import PySimpleGUI as sg

def filtering(spectrum, flag):
    #スペクトルを逆FFTしてインパルス応答へ
    h = np.real(np.fft.ifft(spectrum))

    #円状シフトと窓関数法
    shifted_h = np.roll(h,int(tap/2))
    shifted_h = shifted_h * np.hanning(tap)

    #インパルス応答とノイズ入りデータの畳み込み
    result = np.convolve(data.astype(float),shifted_h)

    #出力
    if flag == 0:
        print("noob")
    elif flag == 1:
        sf.write(f"~{fh}hz.wav",result,fs)
    elif flag == 2:
        sf.write(f"{fl}hz~.wav",result,fs)
    elif flag == 3:
        sf.write(f"{fl}~{fh}hz.wav",result,fs)
    
    return

flag = 0
fl = 0
fh = 0
tap = 2048
#GUI生成
layout = [[sg.Text("音声データパス"),sg.Input("snow_noise.mp3",key = "path")],
          [sg.Radio("LPF","1",key = "LPF"),sg.Radio("HPF","1",key = "HPF"),sg.Radio("BPF","1",key = "BPF")],
          [sg.Text("低周波遮断周波数"),sg.Input("0",key = "fl")],
          [sg.Text("高周波遮断周波数"),sg.Input("7000",key = "fh")],
          [sg.Button("Go")]]
window = sg.Window("フィルター",layout)

#イベントループ
while True:
    event, values = window.read()

    if event == "Go":
        #音声取得
        data, fs = lib.load(values["path"])

        #スペクトル生成
        kc_l = int(int(values["fl"]) / fs * tap)
        kc_h = int(int(values["fh"]) / fs * tap)
        spectrum = np.zeros(tap)
        if values["LPF"]:
            flag = 1
            fh = values["fh"]
            spectrum[0:kc_h] = 1.0
            spectrum[tap - kc_h + 1:] = 1.0

            filtering(spectrum,flag)
        elif values["HPF"]:
            flag = 2
            fl = values["fl"]
            spectrum[kc_l:tap - kc_l] = 1.0

            filtering(spectrum, flag)
        elif values["BPF"]:
            flag = 3
            fl = values["fl"]
            fh = values["fh"]
            spectrum[kc_l:kc_h] = 1.0
            spectrum[tap - kc_h + 1:tap - kc_l] = 1.0

            filtering(spectrum, flag)
    
    if event == sg.WINDOW_CLOSED:
        break
