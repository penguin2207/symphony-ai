import argparse
import datetime
import numpy as np
import sys
import math
import itertools
from scipy.io import wavfile
import scipy.signal as sig
import matplotlib.pyplot as plt
import noisereduce as nr
import pyloudnorm as pyln

from aubio import source, onset

from scipy import signal
from scipy.fft import fftshift

from scipy.io.wavfile import write, read

from midiutil import MIDIFile

import soundfile as sf

from aubio_onsets import *


def wav_to_arr(fname):
    sr, data =read(fname)
    print("File read in "+ str(fname)+" with sample rate "+str(sr))
    return sr, data.astype(np.float64)


def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = sig.butter(order, normalCutoff, btype='low', analog = True)
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y


def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20*np.log10(abs(np.where(sd == 0, 0, m/sd)))


guitar_notes = [82.41,	87.31,	92.50,	98.00,	103.8,	110.0,	116.5,	123.5,	130.8,	138.6,	146.8,	155.6,	164.8,	174.6,	185.0,	196.0,	207.7,	220.0,	233.1,	246.9,	261.6,	277.2,	293.7,	311.1,	329.6,	349.2,	370.0,	392.0,	415.3,	440.0,	466.2,	493.9,	523.3,	554.4,	587.3,	622.3,	659.3,	698.5,	740.0,	784.0,	830.6,	880.0,	932.3,	987.8,	1047,	1109,	1175,	1245,	1319]

def autoCor(data, W, t, lag):
    arr1 = data[t: t+W]
    arr2 = data[lag+t : lag + t + W]

    difference = arr1.shape[0] - arr2.shape[0]

    return np.dot(arr1, np.concatenate([arr2, np.zeros((difference))]))


def DF(data, W, t, lag):
    return autoCor(data, W, t, 0) + autoCor(data, W, t+lag, 0) - (2 * autoCor(data, W, t, lag))


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

def hz_to_midi(f):
    return 12*(math.log(f/220.0)/math.log(2))+57

def to_midi(p, t, d, mid_out):
    notes = list(map(hz_to_midi, p))

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
        MyMIDI.addNote(track, channel, int(n), t[i], d[i], volume, None)

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
    win_s = 256                 # fft size
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
            #print("%f" % (o.get_last_s()))
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
    parser.add_argument("midi_out", type=str, help='midi out file')
    args = parser.parse_args()

    fs = 44100

    ## Normalize audio
    
    data_sf, rate = sf.read(args.wav_in)

    # peak normalize audio to -1 dB
    peak_normalized_audio = pyln.normalize.peak(data_sf, -1.0)

    # measure the loudness first 
    meter = pyln.Meter(fs) # create BS.1770 meter
    loudness = meter.integrated_loudness(data_sf)
    
    # loudness normalize audio to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(data_sf, loudness, -12.0)
    
    write("normalized.wav", fs,loudness_normalized_audio)

    rate, data = wav_to_arr("normalized.wav")

    if len(data.shape) > 1 and data.shape[1] > 1: #mono mixdown
        data = data.mean(axis=1)

    cutOff = 2000 #cutoff frequency in rad/s
    order = 20 #order of filter
    data_filt = butter_lowpass_filter(data, cutOff, fs, order)
    
    window_size = 0.025
    window = int(window_size*fs)
    pitches = []
    maxs = []
    on_offs = []
    
    for j in range(data_filt.shape[0] // (window+3)):
        #print("Window: ", (j+1), "/", (data_filt.shape[0] // (window+3)))
        #print(data_filt[j*window:(j+1)*window])
        loudness = max(data_filt[j*window:(j+1)*window])
        #print(loudness)
        if(loudness>(.6*10**(-22))):
           #print("Note On")
           on_offs.append(1)
        else: 
            #print("Note Off")
            on_offs.append(0)
        maxs+=(window)*[loudness]
        pitches.append(augmented_detect_pitch_CMNDF(data_filt, window, j*window, fs))
    
    
    
    t = [ float(t) * ((window)) / fs for t in range(len(data_filt)) ]
    pad_size_maxs=int((len(t)-len(maxs)))
    maxs = np.pad(maxs, (0, pad_size_maxs), 'constant')

    # plt.plot(t,data_filt, label = "orig_signal")
    # plt.plot(t,maxs, '-r', label = "means")
    # plt.legend()
    # plt.show()

    t_pitches = [ float(t) * ((window)) / fs for t in range(len(pitches)) ]
    pad_size_pitches=int((len(t_pitches)-len(pitches)))
    pitches = np.pad(pitches, (0, pad_size_pitches), 'constant')
    # plt.plot(t_pitches, pitches, label = "pitches")
    # y_min =plt.axis()[2]
    # y_max =plt.axis()[3]
    onsets = get_onsets(args.wav_in)
    ons_arr = np.array(onsets)
    ons_div = ons_arr/float(fs)
    onsets = ons_div.tolist()
    # for stamp in onsets:
    #     plt.plot([stamp, stamp], [y_min, y_max], '-r')
    #     stamp /= float(fs)
    # plt.legend()
    # plt.show()


    p=[]
    thresh = 10
    for i, val in enumerate(pitches):
        if(i!=0 and i<len(pitches)-1):
            temp=min(abs(pitches[i]-pitches[i-1]), abs(pitches[i+1]-pitches[i]))
            #print(temp)
            if(float(temp)>float(thresh)):
                p.append(float((pitches[i-1]+pitches[i+1]) / 2.0))
            else:
                p.append(pitches[i])
        else:
            p.append(pitches[i])
    

    p = np.pad(p, (0, pad_size_pitches), 'constant')
    
    # plt.plot(t_pitches, p, label = "pitches_filt")
    # y_min =plt.axis()[2]
    # y_max =plt.axis()[3]
    # for i, stamp in enumerate(onsets):
    #     if(i==0):
    #         plt.plot([stamp, stamp], [y_min, y_max], '-r', label = "note_onsets")
    #     else:
    #         plt.plot([stamp, stamp], [y_min, y_max], '-r')
    #     stamp /= float(fs)
        
    # plt.plot(t_pitches, [i * y_max for i in on_offs], '-g', label = "note_on")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (hz)")
    # plt.title("Pitch Detection Output")
    # plt.legend()
    # plt.show()

    times = []
    out_notes = []
    durs = []
    cur_p=0


    for i, ons in enumerate(onsets):
        close_time = closest(t_pitches, ons)
        if(close_time>=0.01):
            ind = t_pitches.index(close_time,0,len(t_pitches))
            if(i==len(onsets)-1):
                if True in (ele == 1 for ele in on_offs[ind:ind+10]):
                    dur = 0
                    try:
                        dur = abs(t_pitches[on_offs.index(0,ind, len(on_offs)-1)]-closest(t_pitches, onsets[i]))
                    except ValueError as ve:
                        dur = abs(t_pitches[-1]-closest(t_pitches, onsets[i]))
                    if(dur >= .1):
                        out_notes.append(sum(p[ind:ind+20])/20)
                        times.append(t_pitches[ind])
                        durs.append(dur)
            else:
                if True in (ele == 1 for ele in on_offs[ind:ind+10]):
                    
                    dur=0
                    close_plus1_time = closest(t_pitches, onsets[i+1])
                    close_ind_plus1 = t_pitches.index(close_plus1_time,0,len(t_pitches)-1)
                    try:
                        try_off_ind = on_offs.index(0,ind, len(on_offs)-1)
                    except ValueError as ve:
                        try_off_ind=0
                    if(try_off_ind!=0):
                        if(close_ind_plus1<=try_off_ind):
                            dur = abs(closest(t_pitches, onsets[i+1])-closest(t_pitches, onsets[i]))
                        else:
                            dur = abs(t_pitches[try_off_ind]-closest(t_pitches, onsets[i]))
                        if(dur>.1):
                            times.append(t_pitches[ind])
                            out_notes.append(sum(p[ind:ind+20])/20)
                            durs.append(dur)
                            
                    else:
                        dur = abs(closest(t_pitches, onsets[i+1])-closest(t_pitches, onsets[i]))
                        if(dur>.1):
                            times.append(t_pitches[ind])
                            out_notes.append(sum(p[ind:ind+20])/20)
                            durs.append(dur)
                    
                    




    # for i,pitch in enumerate(p):
        
    #     added=False
    #     for ons in onsets:
    #         if abs(t_pitches[i]-ons)<=0.01 and on_offs[i] == 1:
    #             out_notes.append(pitch)
    #             times.append(t_pitches[i])
    #             added = True
            
    #     if(pitch!=cur_p):
    #         if(on_offs[i] == 1 and not added):
    #             out_notes.append(pitch)
    #             times.append(t_pitches[i])
    #         elif(i!=0):
    #             if(on_offs[i-1] == 1 and not added):
    #                 out_notes.append(pitch)
    #                 times.append(t_pitches[i])
    #         elif(i<len(p)-1):
    #             if(on_offs[i+1] == 1 and not added):
    #                 out_notes.append(pitch)
    #                 times.append(t_pitches[i])
    #         cur_p=pitch
    #     prev_on_off = on_offs[i]

    midi_score = to_midi(out_notes, times, durs, args.midi_out)

    # print("Now: ")
    # print(now.strftime("%Y-%m-%d %H:%M:%S"))
    # print("Later: ")
    # print(later.strftime("%Y-%m-%d %H:%M:%S"))
    # print("Time: %i" % difference.microseconds)
    # print(args.wav_in)

    # plt.plot(data, label = "orig signal")

    print("SNR: "+str(signaltonoise_dB(data_filt)))
    if(signaltonoise_dB(data_filt)> -25):
        print("!!!ERRORS LIKELY, SNR TOO HIGH!!!")