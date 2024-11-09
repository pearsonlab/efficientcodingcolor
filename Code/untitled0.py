#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:08:52 2024

@author: david
"""

import skvideo
import skvideo.io
from scipy.ndimage import gaussian_filter
from scipy.signal import lfilter
from scipy import signal
import numpy as np


def convolve2d_video(video, sigma):
    video_conv = np.zeros(video.shape)
    for t in range(video.shape[0]):
        for c in range(video.shape[3]):
            video_tc = video[t,:,:,c]
            video_conv[t,:,:,c] = gaussian_filter(video_tc, sigma)
    return video_conv


def write_video(video):
    skvideo.io.vwrite("testvideo.mp4", video*255)
    
def temporal_filter_video(video):
    b, a = signal.butter(3, 0.05)
    video_conv = np.zeros(video.shape)
    for s1 in range(video.shape[1]):
        for s2 in range(video.shape[2]):
            for c in range(video.shape[3]):
                video_ssc = video[:,s1,s2,c]
                video_conv[:,s1,s2,c] = lfilter(b,a,video_ssc)
                
    return video_conv
    