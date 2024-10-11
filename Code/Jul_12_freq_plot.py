#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:09:42 2024

@author: david
"""

colors = ['r', 'g', 'b']

size = 100
max_range= int(size/2)
freqs = np.fft.fftfreq(size)
for c in range(3):
    plt.plot(np.log10(freqs[0:max_range]), np.log10(np.mean(psd_2D, axis = 1))[c, 0:max_range], color = colors[c])
    plt.plot(np.log10(freqs[0:max_range]), np.log10(np.mean(psd_2D, axis = 2))[c, 0:max_range], color = colors[c], linestyle = "dashed")
plt.xlabel("Log(Frequency)", size = 30)
plt.ylabel("Log(Power)", size = 30)