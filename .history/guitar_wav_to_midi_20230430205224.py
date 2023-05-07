import argparse
import librosa
import numpy as np
import sys
import math
import itertools
from scipy.io import wavfile
import scipy.signal as sig
import matplotlib.pyplot as plt

import pyloudnorm as pyln

from aubio import source, onset

from scipy import signal
from scipy.fft import fftshift

from scipy.ndimage import gaussian_filter1d

from scipy.io.wavfile import write, read

from midiutil import MIDIFile

import soundfile as sf

from aubio_onsets import *


def wav_to_arr(fname):
    sr, data =read(fname)
    print("File read in "+ str(fname)+" with sample rate "+str(sr))
    return sr, data.astype(np.float64)


def nco(fcw, acw, sr, phw):
    phase = 0
    phase_result = []
    ind=0
    for fcw_samp in fcw:
        new_ph=2*np.pi*phw[ind] * 1/sr
        ph_step = 2*np.pi* fcw_samp * 1/sr
        delta_ph = ph_step+new_ph
        phase += delta_ph
        phase_result.append(phase)
        ind+=1
    return acw*np.cos(phase_result)


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = sig.butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y



#print sticker_data.ps1_dxdt2






def gaussian_filt(sr, data):
    window_dur=0.01
    nsamps = int(data.shape[0]/10.0)
    coeff = sig.gaussian(nsamps,50)
    return sig.filtfilt(coeff,np.sum(coeff),data)


def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))


guitar_notes = [82.41,	87.31,	92.50,	98.00,	103.8,	110.0,	116.5,	123.5,	130.8,	138.6,	146.8,	155.6,	164.8,	174.6,	185.0,	196.0,	207.7,	220.0,	233.1,	246.9,	261.6,	277.2,	293.7,	311.1,	329.6,	349.2,	370.0,	392.0,	415.3,	440.0,	466.2,	493.9,	523.3,	554.4,	587.3,	622.3,	659.3,	698.5,	740.0,	784.0,	830.6,	880.0,	932.3,	987.8,	1047,	1109,	1175,	1245,	1319]

def autoCor(data, W, t, lag):
    #print("custom: " + str((np.sum(data[t: t+W] * data[lag+t : lag + t + W]))))
    #print("sig: " + str((sig.correlate(data[t: t+W], data[lag+t : lag + t + W], method = 'fft'))))
    arr1 = data[t: t+W]
    arr2 = data[lag+t : lag + t + W]

    difference = arr1.shape[0] - arr2.shape[0]

    return np.dot(arr1, np.concatenate([arr2, np.zeros((difference))]))
    
    #return np.sum(data[t: t+W] * data[lag+t : lag + t + W])
    
    #return sig.correlate(data[t: t+W], data[lag+t : lag + t + W], method = 'fft')[0]


def DF(data, W, t, lag):
    return autoCor(data, W, t, 0) + autoCor(data, W, t+lag, 0) - (2 * autoCor(data, W, t, lag))

def CMNDF(data, W, t, lag):
    if lag == 0:
        return 1
    return DF(data, W, t, round(lag)) / np.sum([DF(data, W, t, round(j)) for j in guitar_notes[0:guitar_notes.index(lag)] ]) * round(lag)
#for j in guitar_notes[0:guitar_notes.index(lag)]


def memo_CMNDF(f, W, t, lag_max):
    running_sum = 0
    vals = []
    for lag in range(0, lag_max):
        if lag == 0:
            vals.append(1)
            running_sum += 0
        else:
            running_sum += DF(f, W, t, lag)
            vals.append(DF(f, W, t, lag) / running_sum * lag)
    return vals

def pitch_detect(f, W, t, fs, thresh=0.1):
    vals_from_CMNDF = [CMNDF(f, W, t, val) for val in guitar_notes]
    sample = None
    # print("Detecting Pitch 1")
    for i, val in enumerate(vals_from_CMNDF):
        if val < thresh:
            sample = round(guitar_notes[i]) + round(guitar_notes[0])
            break
    # sprint("Detecting Pitch 2")
    if sample is None:
        sample = np.argmin(vals_from_CMNDF) + round(guitar_notes[0])
    if sample == 0:
        return 0
    return fs/sample

def augmented_detect_pitch_CMNDF(f, W, t, sample_rate, thresh=0.1):  # Also uses memoization
    CMNDF_vals = memo_CMNDF(f, W, t, guitar_notes[-1])[round(guitar_notes[0]):]
    sample = None
    for i, val in enumerate(CMNDF_vals):
        if val < thresh:
            sample = i + round(guitar_notes[0])
            break
    if sample is None:
        sample = np.argmin(CMNDF_vals) + round(guitar_notes[0])
    return sample_rate / (sample + 1)


def closest(lst, K):
     
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]

def to_midi(p, t, mid_out):
    notes = []
    for val in p:
        notes = librosa.hz_to_midi(p)
    track    = 0
    channel  = 0
    time     = 0   # In beats
    duration = 1   # In beats
    tempo    = 60  # In BPM
    volume   = 100 # 0-127, as per the MIDI standard

    MyMIDI = MIDIFile(1) # One track, defaults to format 1 (tempo track
                         # automatically created)
    MyMIDI.addTempo(track,time, tempo)

    for i,n in enumerate(notes):
        MyMIDI.addNote(track, channel, int(n), t[i], duration, volume)

    with open(mid_out, "wb") as output_file:
        MyMIDI.writeFile(output_file)
    


def get_onset_times(file_path, fs):
    window_size = 1024 # FFT size
    hop_size = window_size // 4

    sample_rate = 0
    src_func = source(file_path, sample_rate, hop_size)
    sample_rate = src_func.samplerate
    onset_func = onset('default', window_size, hop_size)
    
    duration = float(src_func.duration) / src_func.samplerate

    onset_times = [] # seconds
    while True: # read frames
        samples, num_frames_read = src_func()
        if onset_func(samples):
            onset_time = onset_func.get_last_s()
            if onset_time < duration:
                onset_times.append(onset_time)
            else:
                break
        if num_frames_read < hop_size:
            break
    
    return onset_times

def onsets(filename, fs):
    win_s = 512                 # fft size
    hop_s = win_s // 2          # hop size

    s = source(filename, fs, hop_s)
    o = onset("default", win_s, hop_s, fs)

    # list of onsets, in samples
    onsets = []

    # storage for plotted data
    desc = []
    tdesc = []
    allsamples_max = np.zeros(0,)
    downsample = 2  # to plot n samples / hop_s

    # total number of frames read
    total_frames = 0
    while True:
        samples, read = s()
        if o(samples):
            print("%f" % (o.get_last_s()))
            onsets.append(o.get_last())
        # keep some data to plot it later
        new_maxes = (abs(samples.reshape(hop_s//downsample, downsample))).max(axis=0)
        allsamples_max = np.hstack([allsamples_max, new_maxes])
        desc.append(o.get_descriptor())
        tdesc.append(o.get_thresholded_descriptor())
        total_frames += read
        if read < hop_s: break
    return onsets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_in", type=str, help='path to example wav')
    parser.add_argument("wav_out", type=str, help='path to wav out')
    parser.add_argument("midi_out", type=str, help='midi out file')
    args = parser.parse_args()
    #main(args.wav_in, args.wav_out)

    #data, fs = librosa.load(args.wav_in)
    #data = librosa.to_mono(data)

    ## Normalize audio

    data_sf, rate = sf.read(args.wav_in)

    # peak normalize audio to -1 dB
    peak_normalized_audio = pyln.normalize.peak(data_sf, -1.0)

    # measure the loudness first 
    meter = pyln.Meter(rate) # create BS.1770 meter
    loudness = meter.integrated_loudness(data_sf)
    
    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data_sf, loudness, -12.0)

    write("normalized.wav", rate,loudness_normalized_audio)

    fs, data = wav_to_arr("normalized.wav")

    if len(data.shape) > 1 and data.shape[1] > 1: #mono mixdown
        data = data.mean(axis=1)

    cutOff = 2000 #cutoff frequency in rad/s
    order = 20 #order of filter
    data_filt = butter_lowpass_filter(data, cutOff, fs, order)
    write("filtered.wav", fs,data_filt)

    

    #pad = np.zeros(100)
    #data_filt_pad = np.append(pad, loudness_normalized_audio)
    data_filt_pad = loudness_normalized_audio
    print(signaltonoise_dB(data_filt_pad))
    
    # proc_fft(args.wav_in, args.wav_out)
    
    window_size = 0.025
    window = int(window_size*fs)
    pitches = []
    maxs = []

    
    for j in range(data_filt_pad.shape[0] // (window+3)):
        print("Window: ", (j+1), "/", (data_filt_pad.shape[0] // (window+3)))
        #print(data_filt_pad[j*window:(j+1)*window])
        loudness = max(data_filt_pad[j*window:(j+1)*window])
        print(loudness)
        if(loudness>.15):
           print("Note On")
        else: 
            print("Note Off")
        maxs+=(window)*[loudness]
        pitches.append(augmented_detect_pitch_CMNDF(data_filt_pad, window, j*window, fs))
        #closest(guitar_notes, pitch_detect(data, window, i*window, fs))
    
    
    
    t = [ float(t) * ((window)) / fs for t in range(len(data_filt_pad)) ]
    pad_size_maxs=int((len(t)-len(maxs)))
    maxs = np.pad(maxs, (0, pad_size_maxs), 'constant')

    plt.plot(t,data_filt_pad, label = "orig_signal")
    plt.plot(t,maxs, '-r', label = "means")
    plt.legend()
    plt.show()

    t_pitches = [ float(t) * ((window)) / fs for t in range(len(pitches)) ]
    pad_size_pitches=int((len(t_pitches)-len(pitches)))
    pitches = np.pad(pitches, (0, pad_size_pitches), 'constant')
    plt.plot(t_pitches, pitches, label = "pitches")
    y_min =plt.axis()[2]
    y_max =plt.axis()[3]
    onsets = get_onsets(args.wav_in)
    for stamp in onsets:
        stamp /= float(fs)
        plt.plot([stamp, stamp], [y_min, y_max], '-r')
    plt.legend()
    plt.show()


    p=[]
    thresh = 10
    for i, val in enumerate(pitches):
        if(i!=0 and i<len(pitches)-1):
            temp=min(abs(pitches[i]-pitches[i-1]), abs(pitches[i+1]-pitches[i]))
            print(temp)
            if(float(temp)>float(thresh)):
                p.append(closest(guitar_notes, float((pitches[i-1]+pitches[i+1]) / 2.0)))
            else:
                p.append(closest(guitar_notes, pitches[i]))
        else:
            p.append(closest(guitar_notes, pitches[i]))
    

    p = np.pad(p, (0, pad_size_pitches), 'constant')
    plt.plot(t_pitches, p, label = "pitches_filt")
    y_min =plt.axis()[2]
    y_max =plt.axis()[3]
    onsets = get_onsets(args.wav_in)
    for stamp in onsets:
        stamp /= float(fs)
        plt.plot([stamp, stamp], [y_min, y_max], '-r')
    plt.legend()
    plt.show()

    times = []
    for stamp in onsets:



    
    midi_score = to_midi(pitches, times, args.midi_out)
    print(args.wav_in)

    #plt.plot(data, label = "orig signal")
    #t = [*range(0,round(len(data_filt_pad)/fs * 1000),1)]
    

    # 
    # for stamp in onsets:
    #     stamp /= float(fs)
    #     stamp *= 100
    #     plt.plot([stamp, stamp], [-1., 1.], '-r')

    print("SNR: "+str(signaltonoise_dB(data_filt_pad)))
    if(signaltonoise_dB(data_filt_pad)> -25):
        print("!!!ERRORS LIKELY, SNR TOO HIGH!!!")

    
    # f, t, Sxx = signal.spectrogram(data, fs)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    # proc_fft(args.wav_in, args.wav_out)

