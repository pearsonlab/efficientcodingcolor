#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 17:00:50 2024

@author: david
"""
import numpy
size = 180
n_TFs = 125
n_colors = 3
n_bins = 200
freqs1d = numpy.fft.fftfreq(size*2)[0:int(size)]
y_freqs = numpy.repeat(numpy.repeat(numpy.repeat(freqs1d[numpy.newaxis,numpy.newaxis,:, numpy.newaxis], n_TFs, axis = 0), n_colors, axis = 1), freqs1d.shape[0], axis = 3)
x_freqs = numpy.repeat(numpy.repeat(numpy.repeat(freqs1d[numpy.newaxis,numpy.newaxis,numpy.newaxis,:], n_TFs, axis = 0), n_colors, axis = 1), freqs1d.shape[0], axis = 2)

freqs = x_freqs**2 + y_freqs**2
test = numpy.random.normal(size=[125,3,180,180])
bins = numpy.linspace(numpy.min(freqs), numpy.max(freqs), n_bins)
digitized = numpy.digitize(freqs, bins)
bin_means = [test[digitized == i].reshape([test.shape[0],-1, n_colors]).mean(axis=1)for i in range(1, len(bins)-1)]


#Tests: If the two are the same then you are fine
test[digitized == 2].reshape([test.shape[0], 3, -1])[15,1,:].mean()
test[15,1,:,:][digitized[15,1,:,:] == 2].mean()

#This tests suggests test.shape[0], 3, -1 is the right order
