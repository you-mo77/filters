import numpy as np
import librosa as lib
import soundfile as sf
import pyaudio as pa
import time 
from scipy import signal

num = 6

#音声取得
#data, fs = lib.load(f"crystalized_{num}.wav",mono=False, sr=48000)
data, fs = lib.load("crystalized_.5.wav", mono=False, sr=48000)

#初期化
first_callback = True
buffer_size = 48000
ex_buffer = np.zeros((2,buffer_size))
for_filter_array = np.zeros((2,buffer_size * 2))

#出力配列(これにフィルタ後のデータをたしてく)
total1 = np.zeros((2,buffer_size))
total2 = np.zeros((2,buffer_size))
total3 = np.zeros((2,buffer_size))

#スタート位置
start_pos = 0

#コールバック関数
def callback(frame_count,fc):
    global total1,total2,total3
    global start_pos, data, fs
    global ex_buffer
    global first_callback
    
    # 切り出す(ステレオ->2行frame_count列　)
    output_data = data[:, start_pos:(start_pos + frame_count)]

    #フィルタリング用配列(前回バッファと今回バッファ結合->クロスフェード用に山なりに変化させる)
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

#main関数
def main():

    #カットオフ周波数forスペクトル(最大周波数はサンプル周波数/2)
    fc = [0,700,7000,23999]

    #pyaudioインスタンス
    p = pa.PyAudio()

    #ストリーム
    print(f"opening {data.shape[0]} channels")
    stream = p.open(format=pa.paFloat32,
                    channels=2,
                    rate=fs,
                    output=True,
                    stream_callback=lambda a,b,c,d:callback(b,fc),
                    frames_per_buffer=buffer_size)

    while stream.is_active():
        time.sleep(0.1)
    
    #音声ファイル出力(テスト用　実際はいらない)
    sf.write(f"{num}_output1.new.wav",total1.T,fs,format="wav")
    sf.write(f"{num}_output2.new.wav",total2.T,fs,format="wav")
    sf.write(f"{num}_output3.new.wav",total3.T,fs,format="wav")

    #終了処理
    stream.close()
    p.terminate()

    return 

#main
if __name__ == "__main__":
    main()
