import socket               
from threading import Thread
import threading
import struct
import time
import signal
import sys
# from guitar_wav_to_midi import midi_convert
import os
import _thread
from time import sleep
import io
import numpy as np

host = "172.26.177.232" #ESP32 IP in local network
port = 80 #ESP32 Server Port    
sound_recording=False
record_data=False

def get_input():
    response=''
    global sound_recording
    global record_data
    while(1):
        if sound_recording:
            try:
                response=input("Do you want to switch off recording? \n")
            except:
                _thread.interrupt_main()
            if response.strip().lower()=='yes':
                sound_recording=False
                record_data=True
        else:
            try:
                response=input("Do you want to start recording? \n")
            except:
                _thread.interrupt_main()
            if response.strip().lower()=='yes':
                sound_recording=True
        time.sleep(2)

def write_header(_bytes, _nchannels, _sampwidth, _framerate):
    WAVE_FORMAT_PCM = 0x0001
    initlength = len(_bytes)
    bytes_to_add = b'RIFF'
    _nframes = initlength // (_nchannels * _sampwidth)
    _datalength = _nframes * _nchannels * _sampwidth
    bytes_to_add += struct.pack('<L4s4sLHHLLHH4s',
        36 + _datalength, b'WAVE', b'fmt ', 16,
        WAVE_FORMAT_PCM, _nchannels, _framerate,
        _nchannels * _framerate * _sampwidth,
        _nchannels * _sampwidth,
        _sampwidth * 8, b'data')
    bytes_to_add += struct.pack('<L', _datalength)
    return bytes_to_add + _bytes

def get_recording_data():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    global sound_recording
    global record_data
    data=[]
    while(1):
        if record_data:
            if os.path.exists('current_sound.wav'):
                os.remove('current_sound.wav')
            wav_file=write_header(bytes(data),2,2,22050)
            with open('current_sound.wav', mode='wb') as f:
                f.write(wav_file)
            print('file write done')
            data=[]
            record_data=False
        if sound_recording:
            try:
                data+=sock.recv(1024)
            except:
                _thread.interrupt_main()
        else:
            try:
                garb=sock.recv(1024)
            except:
                _thread.interrupt_main()
    sock.close()

def signal_handler(signum, frame):
    sys.exit()

signal.signal(signal.SIGINT, signal_handler)
input_thread=Thread(target=get_input)
recording_thread=Thread(target=get_recording_data)
input_thread.daemon=True
recording_thread.daemon=True
input_thread.start() 
recording_thread.start()
while(True):
    sleep(2)