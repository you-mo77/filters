"""
play.py

Stream a wav file to the default output device.
Supports specifying backend, device and block size.

Requires soundfile
    pip3 install soundfile

http://pysoundfile.readthedocs.io/
"""
import argparse
import time

import soundfile as sf
from pysoundio import (
    PySoundIo,
    SoundIoFormatFloat32LE,
)


class Player:

    def __init__(self, infile, backend=None, output_device=None, block_size=None):

        data, rate = sf.read(
            infile,
            dtype='float32',
            always_2d=True
        )
        self.idx = 0
        self.stream = data.tobytes()
        self.block_size = block_size

        self.total_blocks = len(data)
        self.timer = self.total_blocks / float(rate)

        self.num_channels = data.shape[1]
        self.sample_size = data.dtype.itemsize

        self.pysoundio = PySoundIo(backend=backend)
        self.pysoundio.start_output_stream(
            device_id=output_device,
            channels=self.num_channels,
            sample_rate=rate,
            block_size=self.block_size,
            dtype=SoundIoFormatFloat32LE,
            write_callback=self.callback
        )

        print('%s:\n' % infile)
        print(' Channels: %d' % data.shape[1])
        print(' Sample rate: %dHz' % rate)
        print('')

    def close(self):
        self.pysoundio.close()

    def callback(self, data, length):
        num_bytes = length * self.sample_size * self.num_channels
        data[:] = self.stream[self.idx:self.idx+num_bytes]
        self.idx += num_bytes


def get_args():
    parser = argparse.ArgumentParser(
        description='PySoundIo audio player example',
        epilog='Play a wav file over the default output device'
    )
    parser.add_argument('infile', help='WAV output file name')
    parser.add_argument('--backend', type=int, help='Backend to use (optional)')
    parser.add_argument('--blocksize', type=int, default=4096, help='Block size (optional)')
    parser.add_argument('--device', type=int, help='Output device id (optional)')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    player = Player(args.infile, args.backend, args.device, args.blocksize)
    print('Playing...')
    print('CTRL-C to exit')

    try:
        time.sleep(player.timer)
    except KeyboardInterrupt:
        print('Exiting...')

    player.close()


if __name__ == '__main__':
    main()
