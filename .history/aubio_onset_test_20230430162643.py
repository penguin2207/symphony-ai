#! /usr/bin/env python

import sys
from aubio import onset, source
from numpy import hstack, zeros

win_s = 512                 # fft size
hop_s = win_s // 2          # hop size

if len(sys.argv) < 2:
    print("Usage: %s <filename> [samplerate]" % sys.argv[0])
    sys.exit(1)

filename = sys.argv[1]

samplerate = 0
if len( sys.argv ) > 2: samplerate = int(sys.argv[2])

s = source(filename, samplerate, hop_s)
samplerate = s.samplerate
o = onset("default", win_s, hop_s, samplerate)

# list of onsets, in samples
onsets = []

# storage for plotted data
desc = []
tdesc = []
allsamples_max = zeros(0,)
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
    allsamples_max = hstack([allsamples_max, new_maxes])
    desc.append(o.get_descriptor())
    tdesc.append(o.get_thresholded_descriptor())
    total_frames += read
    if read < hop_s: break




if 1:
    # do plotting
    import matplotlib.pyplot as plt
    allsamples_max = (allsamples_max > 0) * allsamples_max
    allsamples_max_times = [ float(t) * hop_s / downsample / samplerate for t in range(len(allsamples_max)) ]

    plt.rc('lines',linewidth='.8')
    plt1.plot(allsamples_max_times,  allsamples_max, '-b')
    plt1.plot(allsamples_max_times, -allsamples_max, '-b')
    for stamp in onsets:
        stamp /= float(samplerate)
        plt1.plot([stamp, stamp], [-1., 1.], '-r')
    plt1.axis(xmin = 0., xmax = max(allsamples_max_times) )
    plt.xlabel('time (s)')
    #plt.savefig('/tmp/t.png', dpi=200)
    plt.show()