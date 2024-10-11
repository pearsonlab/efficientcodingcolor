#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:59:38 2024

@author: david
"""

import os
import skvideo
import skvideo.io
import skimage
import numpy as np
import matplotlib.pyplot as plt
import scipy
import moviepy.editor
import time
import torch
np.float = np.float64
np.int = np.int_


global_path = "/home/david/Github/video_color_PSD/"
device = "cuda:1"
#Taken from: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
adobergb2xyz = torch.tensor(np.array([[0.5767309, 0.1855540, 0.1881852],
                            [0.2973769, 0.6273491, 0.0752741],
                            [0.0270343, 0.0706872, 0.9911085]]), device=device)

srgb2xyz = torch.tensor(np.array([[0.4127, 0.3586, 0.1808],
                     [0.2132, 0.7172, 0.0724],
                     [0.0195, 0.1197, 0.9517]]), device=device).type(torch.float32)
                    
#Taken from Colorimetry book, page 305. 
xyz2lms = torch.tensor(np.array([[0.4002,  0.7075, 0.0807],
                        [0.2280, 1.1500, 0.0612],
                        [0,      0,      0.9184]]),device=device).type(torch.float32)

#Taken from the Colorimetry book, page 227. 10 degrees 
rgb2lms = torch.tensor(np.array([[0.192325269,  0.749548882,  0.0675726702],
                       [0.0192290085, 0.940908496,  0.113830196],
                       [0,            0.0105107859, 0.991427669]]),device=device)

#Taken from: http://www.scikit-video.org/stable/examples/io.html
#Reads a YUV video with no conversion

#Takes 2D spectrogram in spacexspace and returns radial freqs and rad power

def check_memory(maximum = 30, print_statement = True):
    total_memory, used_memory, free_memory = map(
    int, os.popen('free -t -m').readlines()[-1].split()[1:])
    if print_statement:
        print("Currently using " + str(used_memory/total_memory*100) + "% of memory")
    if used_memory/total_memory > maximum/100:
        raise Exception("You're using too much memory, stop!")
        
#This function is obsolete. Use load_video_segment instead. 
def load_video(filename, n_frames = 10000, path = global_path):
    check_memory(30)
    return skvideo.io.vread(path + filename, num_frames = n_frames)[:,:,0:500,:]

def load_video_segment(video_clip, frames):
    print("Loading segment")
    fps = video_clip.fps
    check_memory(30, print_statement = False)
    video_segment = []
    frames_array = np.array(range(frames[0], frames[1]))
    for t in frames_array:
        video_segment.append(video_clip.get_frame(t/fps))
    print("Finished loading segment")
    return torch.tensor(np.array(video_segment),device=device)

class video_segment():
    #Takes as input a video object from the moviepy.editor package. From this video, takes a segment from index_range. 
    #This segment has images with 8bits integer values ranging from 0 to 255 in srgb, and converts it to 0 to 1 floats in lms space.  
    #Index_range is a list with 2 values, which represent the start and stop frames of the video slice. 
    def __init__(self, video_clip, frames):
        #index = np.int(np.linspace(index_range[0], index_range[1] - 1, num = index_range[1] - index_range[0]))
        self.video_rgb = load_video_segment(video_clip, frames)
        linear_rgb = self.linearize_srgb()
        print(linear_rgb.shape)
        xyz = torch.matmul(srgb2xyz, torch.swapaxes(linear_rgb, 3, 2))
        print(xyz.shape)
        lms = torch.matmul(xyz2lms, torch.swapaxes(xyz, 0, 2))
        self.video_lms = torch.swapaxes(torch.swapaxes(lms, 0, 2), 1, 2)
        self.video_lms = self.normalize_color(self.video_lms).type(torch.float32)
        self.video = self.video_lms
        
        self.n_colors = self.video_rgb.shape[3]
    
    def linearize_srgb(self):
        linear_rgb = torch.zeros(self.video_rgb.shape,device=device).type(torch.float32)
        img_float = self.video_rgb
        img_float = img_float/torch.tensor(255,device=device)
        above_thresh = img_float > 0.04045
        linear_rgb[~above_thresh] = img_float[~above_thresh]/12.92
        linear_rgb[above_thresh] = ((img_float[above_thresh] + 0.055)/1.055)**2.4
        return linear_rgb
    
    #Normalize each color channel to have mean = 0 and std = 1. Otherwise, S channel just has more power (?)
    def normalize_color(self, video):
        for c in range(3):
            video[:,c,:,:] -= torch.mean(video[:,c,:,:])
            video[:,c,:,:] = video[:,c,:,:]/torch.std(video[:,c,:,:])
        return video
    #Does a 3D PSD on self.video. Only takes real freqs
    def make_psd_3D(self):
        real_spatial_freqs = int(self.video.shape[2]/2)
        real_temporal_freqs = int(self.video.shape[0]/2)
        #psd2D_space = psd2D_space[:, 0:real_spatial_freqs, 0:real_spatial_freqs]

        f3D = np.fft.fftn(self.video, axes = (0,2,3))
        psd3D = abs(f3D)**2
        self.psd3D = psd3D[0:real_temporal_freqs, :, 0:real_spatial_freqs, 0:real_spatial_freqs]
    
    def radial_space_freq(self, space_psd):
        size = space_psd.shape[1]
        freqs = np.fft.fftfreq(size*2)
        freqs_norm = []
        power = []
        for i in range(int(size)):
            for j in range(int(size)):
                freqs_norm.append(np.sqrt(freqs[i]**2 + freqs[j]**2))
                power.append(space_psd[i, j])
        return freqs_norm, power
    
    #Takes as input 1D arrays that represent frequencies (output of radial_space_freq), power and n_bins. Default is 10. 
    def bin_1D_psd(self, freqs, power, n_bins):
        freqs_space_bin = scipy.stats.binned_statistic(freqs, freqs, bins = n_bins)[0]
        power_space_bin = scipy.stats.binned_statistic(freqs, power, bins = n_bins)[0]
        return freqs_space_bin, power_space_bin
    
    #For every temporal frequency, look at the 3D psd for that TF and compute the radial spatial frequencies. Then, bin those radial spatial frequencies in n_bins different bins of the same size. 
    #This returns the log10(STpsd)
    def make_spatiotemporal_psd(self, n_bins):
        n_TFs = self.psd3D.shape[0]
    
        STpsd = []
        for c in range(self.n_colors):
            STpsd_color = []
            for t in range(n_TFs):
                freqs_space, power = self.radial_space_freq(self.psd3D[t,c,:,:])
                #Flaw: I don't do anything with freqs_space_bin... ever. I lose that information
                freqs_space_bin, power_bin = self.bin_1D_psd(freqs_space, power, n_bins = n_bins)
                STpsd_color.append(power_bin)
            STpsd.append(STpsd_color)
        self.STpsd = np.log10(np.array(STpsd))


class PSD():
    def __init__(self, video_name, max_frame = None, path = global_path):
        self.video_clip = moviepy.editor.VideoFileClip(path + video_name)
        if max_frame is None:
            self.n_frames = int(self.video_clip.duration*self.video_clip.fps)
        else:
            self.n_frames = max_frame
    
    #time_bins = how long should each segment be. n_spatial_bins = we need to bin radial frequencies, how many bins do you want in total.
    def average_TS_PSD(self, time_bins, n_spatial_bins):
        current_time = time.time()
        time_points = np.arange(0, self.n_frames, time_bins)
        new = True
        all_PSD = []
        for t in range(time_points.shape[0]-1):
            print("Start:", current_time - time.time())
            check_memory(50, print_statement = False)
            segment = video_segment(self.video_clip, [time_points[t], time_points[t+1]])
            print("Made segment:", current_time - time.time())
            segment.make_psd_3D()
            print("Make_psd_3D:", current_time - time.time())
            segment.make_spatiotemporal_psd(n_spatial_bins) #This is where the power values go through log10
            print("Spatiotemporal_psd", current_time - time.time())
            psd = segment.STpsd
            psd3d = segment.psd3D
            self.last_segment = segment
            all_PSD.append(segment.STpsd)
            if new:
                psd_sum = psd
                psd3d_sum = psd3d
            else:
                psd_sum += psd
                psd3d_sum += psd3d
            print(time_points[t])
        self.PSD = psd_sum/time_points.shape[0]
        self.PSD3D = psd3d_sum/time_points.shape[0]
        self.all_PSD = np.array(all_PSD)
        print("Finish:", current_time - time.time())
    
    
    def import_psd(self, filename, path = global_path):
        self.PSD = np.load(path + filename)
        
        
    def save_psd(self, filename, path = global_path):
        np.save(path + filename, self.PSD)
        
    def log_interp(self, array, n = 100):
        length = array.shape[0]
        y = np.interp(np.logspace(0, np.log10(length), n), np.linspace(0, length-1, length), array)
        return y
    #Default: psd = self.PSD
    def plot_log_spatial_psd(self, psd, n_temp_freqs):
        alphas = np.linspace(0.2, 1, n_temp_freqs)
        fig, ax = plt.subplots(1,1)
        lines = []
        colors = ['red', 'green', 'blue']
        #Need to remove last value because its out of bounds
        for c in range(3):
            index = 0
            for i in np.linspace(0,self.PSD.shape[1],n_temp_freqs)[:-1]:
                y = self.log_interp(self.PSD[c,int(i),:])
                line, = plt.plot(y, label = str(int(i)),  alpha = alphas[index], color = colors[c])
                lines.append(line)
                index += 1
        plt.xlabel("Log(Spatial frequency)", size = 30)
        plt.ylabel("Log(Power)", size = 30)
        ax.legend(handles=lines, title = "Temporal frequency", fontsize = 20)


#fig, ax = plt.subplots(1,3)
#lms = ['L', 'M', 'S']

#for c in range(3):
#    ax[c].imshow(spatiotemporal_psd[c,1:,1:], vmin = 0, vmax = 16)
#    frame1 = plt.gca()
#    frame1.axes.xaxis.set_ticklabels([])
#    frame1.axes.yaxis.set_ticklabels([])
#    ax[c].set_title("PSD for " + lms[c] + " channel", size = 30)
#    ax[c].set_xlabel("Spatial frequency", size = 20)
#    ax[c].set_ylabel("Temporal frequency", size = 20)


#Takes a 1D array and return the interpolated values in log space. 
#diff_psd = spatiotemporal_psd[2,1:,1:] - spatiotemporal_psd[0,1:,1:]


#video_rgb = skimage.color.yuv2rgb(video_yuv)


