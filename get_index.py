import pyaudio as pa

p = pa.PyAudio()

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(info)


for i in range(4):
    print(p.get_host_api_info_by_index(i))
