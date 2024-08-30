#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:59:38 2024

@author: david
"""

import os
import skvideo
import skvideo.io
import cupy as np
import numpy
import matplotlib.pyplot as plt
import scipy
import cupyx.scipy.stats as stats
import moviepy.editor
import time
np.float = np.float64
np.int = np.int_


###NOTE I CHANGED BIN_PSD TO NOT TAKE LOW SPATIAL FREQUENCIES IN X OR Y AS A TEST!!!

#hey = PSD("Nature1_lowres.mp4", [0,10000])
#hey.average_TS_PSD(250,200)

def cuda_mem():
    print(np.get_default_memory_pool().used_bytes()/(1+np.get_default_memory_pool().total_bytes()))
cuda_mem()
global_path = "/home/david/Github/video_color_PSD/"
#Taken from: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
adobergb2xyz = np.array([[0.5767309, 0.1855540, 0.1881852],
                            [0.2973769, 0.6273491, 0.0752741],
                            [0.0270343, 0.0706872, 0.9911085]])

srgb2xyz = np.array([[0.4127, 0.3586, 0.1808],
                     [0.2132, 0.7172, 0.0724],
                     [0.0195, 0.1197, 0.9517]])
                    
#Taken from Colorimetry book, page 305. 
xyz2lms = np.array([[0.4002,  0.7075, 0.0807],
                        [0.2280, 1.1500, 0.0612],
                        [0,      0,      0.9184]])

#Taken from the Colorimetry book, page 227. 10 degrees 
rgb2lms = np.array([[0.192325269,  0.749548882,  0.0675726702],
                       [0.0192290085, 0.940908496,  0.113830196],
                       [0,            0.0105107859, 0.991427669]])

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
    
    
    #Note 1: Run same plot but with log spatial frequency
    #Note 2: Run same analysis with different window sizes and see if we have same result 
#This function is obsolete. Use load_video_segment instead. 
def load_video(filename, n_frames = 10000, path = global_path):
    check_memory(30)
    return skvideo.io.vread(path + filename, num_frames = n_frames)[:,:,0:500,:]

def load_video_segment(video_clip, frames):
    fps = video_clip.fps
    check_memory(30, print_statement = False)
    video_segment = []
    frames_array = np.linspace(int(frames[0]), int(frames[1]), int(frames[1] - frames[0] + 1))
    #frames_array = np.array(range(frames[0], frames[1]))
    for t in frames_array:
        video_segment.append(video_clip.get_frame(t/fps))
    return np.array(video_segment)[:,100:200, 400:500,:]
    #return np.array(video_segment)

class video_segment():
    #Takes as input a video object from the moviepy.editor package. From this video, takes a segment from index_range. 
    #This segment has images with 8bits integer values ranging from 0 to 255 in srgb, and converts it to 0 to 1 floats in lms space.  
    #Index_range is a list with 2 values, which represent the start and stop frames of the video slice. 
    def __init__(self, video_clip, frames, color_means, color_stds):
        #index = np.int(np.linspace(index_range[0], index_range[1] - 1, num = index_range[1] - index_range[0]))
        self.video_rgb = load_video_segment(video_clip, frames)
        linear_rgb = self.linearize_srgb()
        xyz = np.dot(srgb2xyz, np.swapaxes(linear_rgb, 3, 2))
        lms = np.dot(xyz2lms, np.swapaxes(xyz, 0, 2))
        video_lms = np.swapaxes(np.swapaxes(lms, 0, 2), 1, 2)
        #video_lms = self.normalize_color(video_lms)
        self.n_colors = self.video_rgb.shape[3]
        for c in range(self.n_colors):
            video_lms[:,c,:,:] = (video_lms[:,c,:,:] - color_means[c])/color_stds[c]
        self.video = video_lms
        
        
    def linearize_adobergb(self):
        img_float = self.video_rgb.astype(np.float32)
        img_float /= 255
        linear_rgb = img_float**2.2
        return linear_rgb
    
    def linearize_srgb(self):
        linear_rgb = np.zeros(self.video_rgb.shape)
        img_float = self.video_rgb.astype(np.float32)
        img_float /= 255
        above_thresh = img_float > 0.04045
        linear_rgb[np.invert(above_thresh)] = img_float[np.invert(above_thresh)]/12.92
        linear_rgb[above_thresh] = ((img_float[above_thresh] + 0.055)/1.055)**2.4
        return linear_rgb
    
    
    #Normalize each color channel to have mean = 0 and std = 1. Otherwise, S channel just has more power (?)
    def normalize_color(self, video):
        for c in range(3):
            mean = np.mean(video[:,c,:,:])
            std = np.std(video[:,c,:,:])
            video[:,c,:,:] -= mean
            video[:,c,:,:] /= std
            #print(video.shape, mean, std, c)
        return video
    #Does a 3D PSD on self.video. Only takes real freqs
    def make_psd_3D(self):
        real_spatial_freqs = int(self.video.shape[2]/2)
        real_temporal_freqs = int(self.video.shape[0]/2)
        #psd2D_space = psd2D_space[:, 0:real_spatial_freqs, 0:real_spatial_freqs]

        f3D = np.fft.fftn(self.video, axes = (0,2,3))
        psd3D = abs(f3D)**2
        self.f3D = f3D
        self.psd3D = psd3D[0:real_temporal_freqs, :, 0:real_spatial_freqs, 0:real_spatial_freqs]
        
    def make_Cx(self):
        real_spatial_freqs = int(self.video.shape[2]/2)
        real_temporal_freqs = int(self.video.shape[0]/2)
        #psd2D_space = psd2D_space[:, 0:real_spatial_freqs, 0:real_spatial_freqs]

        f3D = np.fft.fftn(self.video, axes = (0,2,3))
    
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
    def bin_1D_psd_old(self, freqs, power, n_bins):
        freqs_np = np.array(freqs).get()
        power_np = np.array(power).get()
        
        freqs_space_bin = scipy.stats.binned_statistic(freqs_np, freqs_np, bins = n_bins)[0]
        power_space_bin = scipy.stats.binned_statistic(freqs_np, power_np, bins = n_bins)[0]
        return np.asarray(freqs_space_bin), np.asarray(power_space_bin)
    
    def bin_psd(self, freqs, power, n_bins):
        bins = numpy.linspace(np.min(freqs), np.max(freqs), n_bins)
        digitized = np.digitize(freqs, bins)
        #self.digitized = digitized #Remove this line later plz 
        bin_means = [power[digitized == i].reshape([self.psd3D.shape[0],self.n_colors, -1]).mean(axis=2)for i in range(1, len(bins)-1)] #I ran tests and this should be the right order
        return np.array(bin_means)
    
    #For every temporal frequency, look at the 3D psd for that TF and compute the radial spatial frequencies. Then, bin those radial spatial frequencies in n_bins different bins of the same size. 
    #This returns the log10(STpsd)
    def make_spatiotemporal_psd(self, n_bins, save_radial_freqs = False, min_freq = 0):
        n_TFs = self.psd3D.shape[0]
        size = self.psd3D.shape[2]
        STpsd = []
        
        freqs = np.fft.fftfreq(size*2)[0:int(size)]
        y_freqs = np.repeat(np.repeat(np.repeat(freqs[np.newaxis,np.newaxis,:, np.newaxis], n_TFs, axis = 0), self.n_colors, axis = 1), freqs.shape[0], axis = 3)
        x_freqs = np.repeat(np.repeat(np.repeat(freqs[np.newaxis,np.newaxis,np.newaxis,:], n_TFs, axis = 0), self.n_colors, axis = 1), freqs.shape[0], axis = 2)
        
        
        freqs_space = np.sqrt(x_freqs**2 + y_freqs**2)
        if save_radial_freqs:
            self.freqs_space = freqs_space #Remove this line if you run out of memory, its meant to be temporary
        
        power_bin = self.bin_psd(freqs_space[:,:,min_freq:-1,min_freq:-1], self.psd3D[:,:,min_freq:-1,min_freq:-1], n_bins)
        self.STpsd = np.log10(power_bin)


class PSD():
    #Frames is an array with 2 values: [frame_min, frame_max]
    def __init__(self, video_name, time_bins, n_spatial_bins, frames = None, path = global_path, means = None):
        
        self.video_clip = moviepy.editor.VideoFileClip(path + video_name)
        if frames is None:
            self.max_frame = int(self.video_clip.duration*self.video_clip.fps)
            self.min_frame = 0
        else:
            self.max_frame = frames[1]
            self.min_frame = frames[0]
        self.time_points = np.arange(self.min_frame, self.max_frame, time_bins)
        self.n_spatial_bins = n_spatial_bins
        self.compute_means()
    def compute_means(self):
        print("Computing mean and std of each color channel")
        means_sum = 0
        stds_sum = 0
        for t in range(self.time_points.shape[0]-1):
            segment = video_segment(self.video_clip, [self.time_points[t], self.time_points[t+1]], [0,0,0], [1,1,1])
            mean = np.mean(segment.video, axis = (0,2,3))
            std = np.std(segment.video, axis = (0,2,3))
            means_sum += mean
            stds_sum += std
            if self.time_points[t]%10000 == 0:
                print(self.time_points[t], means_sum/t, stds_sum/t)
        print("Done computing mean and std of each color channel")
        self.color_means = means_sum/(self.time_points.shape[0]-1)
        self.color_stds = stds_sum/(self.time_points.shape[0]-1)
        
    #time_bins = how long should each segment be. n_spatial_bins = we need to bin radial frequencies, how many bins do you want in total.
    def average_TS_PSD(self):
        current_time = time.time()
        
        new = True
        all_PSD = []
        for t in range(self.time_points.shape[0]-1):
            #print("Start:", time.time() - current_time)
            check_memory(50, print_statement = False)
            segment = video_segment(self.video_clip, [self.time_points[t], self.time_points[t+1]], self.color_means, self.color_stds)
            #print("Made segment:", time.time() - current_time)
            segment.make_psd_3D()
            #print("Make_psd_3D:", time.time() - current_time)
            segment.make_spatiotemporal_psd(self.n_spatial_bins, True) #This is where the power values go through log10
            #print("Spatiotemporal_psd", time.time() - current_time)
            psd = segment.STpsd
            psd3d = segment.psd3D
            self.last_segment = segment
            all_PSD.append(segment.STpsd)
            if new:
                psd_sum = psd
                psd3d_sum = psd3d
                new = False
            else:
                psd_sum += psd
                psd3d_sum += psd3d
            if self.time_points[t]%10000 == 0:
                print(self.time_points[t])
        self.PSD = (psd_sum/self.time_points.shape[0]).get()
        self.PSD3D = psd3d_sum/self.time_points.shape[0]
        self.all_PSD = np.array(all_PSD).get()
        print("Finish:", current_time - time.time())
        cuda_mem()
    
    
    def import_psd(self, filename, path = global_path):
        self.PSD = np.load(path + filename)
        
    def save_psd(self, filename, path = global_path):
        np.save(path + filename, self.PSD)
    
    #To make the plots where frequencies are log-spaced. 
    def log_interp(self, array, n = 100):
        length = array.shape[0]
        y = numpy.interp(numpy.logspace(0, numpy.log10(length), n), numpy.linspace(0, length-1, length), array)
        return y
    #Default: psd = self.PSD
    def plot_log_spatial_psd(self, psd, n_temp_freqs):
        alphas = numpy.linspace(0.2, 1, n_temp_freqs)
        fig, ax = plt.subplots(1,1)
        lines = []
        colors = ['red', 'green', 'blue']
        #Need to remove last value because its out of bounds
        for c in range(3):
            index = 0
            for i in numpy.linspace(0,self.PSD.shape[1],n_temp_freqs)[:-1]:
                y = self.log_interp(self.PSD[:,int(i),c])
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

#Sample code:
#hey = PSD("Nature1_lowres.mp4", 250, 100)
#hey.average_TS_PSD()
#plt.imshow(hey.PSD[:,:,0] - hey.PSD[:,:,2], origin = 'lower', cmap = 'PiYG')
#plt.xlabel("Temporal frequency", size = 30)
#plt.ylabel("Spatial frequency", size = 30)
#plt.title("L - S channels", size = 30)
#hey.plot_log_spatial_psd(hey.PSD3D, 10)
