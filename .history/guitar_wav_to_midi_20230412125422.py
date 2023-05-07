import argparse
import librosa
import numpy as np
from numpy import asarray
import sys
import math
import itertools
from scipy.io import wavfile
import scipy.signal as sig
import matplotlib.pyplot as plt

from scipy import signal
from scipy.fft import fftshift

from scipy.ndimage import gaussian_filter1d

from scipy.io.wavfile import write, read

from midiutil import MIDIFile


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


def proc_fft(wav_in, wav_out):
    # data = a numpy array containing the signal to be processed
    # fs = a scalar which is the sampling frequency of the data
    fs, data = wav_to_arr(wav_in)
    fft_size=2048
    overlap_fac=.25
    hop_size = np.int32(np.floor(fft_size * (1-overlap_fac)))
    pad_end_size = fft_size          # the last segment can overlap the end of the data array by no more than one window size
    total_segments = np.int32(np.ceil(len(data) / np.float32(hop_size)))
    t_max = len(data) / np.float32(fs)
    
    window = np.hanning(fft_size)  # our half cosine window
    inner_pad = np.zeros(fft_size) # the zeros which will be used to double each segment size
    
    proc = np.concatenate((data, np.zeros(pad_end_size)))              # the data to process
    result = np.empty((total_segments, fft_size), dtype=np.float32)    # space to hold the result
    
    for i in range(total_segments):                      # for each segment
        current_hop = hop_size * i                        # figure out the current segment offset
        segment = proc[current_hop:current_hop+fft_size]  # get the current segment
        windowed = segment * window                       # multiply by the half cosine function
        padded = np.append(windowed, inner_pad)           # add 0s to double the length of the data
        spectrum = np.fft.fft(padded) / fft_size          # take the Fourier Transform and scale by the number of samples
        autopower = np.abs(spectrum * np.conj(spectrum))  # find the autopower spectrum
        result[i, :] = autopower[:fft_size]               # append to the results array
    
    result = 20*np.log10(result)          # scale to db
    result = np.clip(result, -40, 200)    # clip values
    # write(wav_out, fs, np.fft.ifft(result))

    img = plt.imshow(result, origin='lower', cmap='jet', interpolation='nearest', aspect='auto')
    plt.show()
    f, t, Sxx = signal.spectrogram(data, fs)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


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

    for n in notes:
        MyMIDI.addNote(track, channel, int(n), time, duration, volume)
        time = time + 1

    with open(mid_out, "wb") as output_file:
        MyMIDI.writeFile(output_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_in", type=str, help='path to example wav')
    parser.add_argument("wav_out", type=str, help='path to wav out')
    parser.add_argument("midi_out", type=str, help='midi out file')
    args = parser.parse_args()
    #main(args.wav_in, args.wav_out)

    #data, fs = librosa.load(args.wav_in)
    #data = librosa.to_mono(data)

    fs, data = wav_to_arr(args.wav_in)

    cutOff = 2000 #cutoff frequency in rad/s
    order = 20 #order of filter
    data_filt = butter_lowpass_filter(data, cutOff, fs, order)

    # proc_fft(args.wav_in, args.wav_out)
    
    pad = np.zeros(100)
    data_pad = np.concatenate([pad, data_filt])
    window = int(25/1000*fs)
    pitches = []
    for i in range(data_pad.shape[0] // (window+3)):
        print("Window: ", (i+1), "/", (data_pad.shape[0] // (window+3)))
        pitches.append(augmented_detect_pitch_CMNDF(data_pad, window, i*window, fs))
        #closest(guitar_notes, pitch_detect(data, window, i*window, fs))

    peaks = librosa.onset.onset_detect(y=data_pad, sr=fs, units='time')

    times = []
    midi_score = to_midi(pitches, times, args.midi_out)

    #plt.plot(data, label = "orig signal")
    plt.plot(pitches, label = "pitches")
    plt.plot(gaussian_filter1d(pitches, 5), label = "pitches_filt")
    plt.plot(peaks, label = "pitches_filt")
    plt.legend()
    plt.show()

    
    # f, t, Sxx = signal.spectrogram(data, fs)
    # plt.pcolormesh(t, f, Sxx, shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()
    #proc_fft(args.wav_in, args.wav_out)

