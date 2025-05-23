#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:59:38 2024

@author: david
"""

import os
#<<<<<<< Updated upstream
import sys
#=======
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg" #This snippet can make it so MoviePy cannot find videos!!!
#>>>>>>> Stashed changes
import skvideo
import skvideo.io
import cupy as cp
import numpy
import pickle
numpy.float_ = numpy.float64
import matplotlib.pyplot as plt
import scipy
import cupyx.scipy.stats as stats
import moviepy.editor
import time
import pandas as pd
from moviepy.video.fx.all import crop
cp.float = cp.float64
cp.int = cp.int_

cp.set_printoptions(legacy='1.13')


video_type = "crop512"
freq_length = 512
Cx_n_bins = 128


names = ['GoProHighlights', 'Incredible_Nature_Scenes', 'our_planet']#, 'GoPro1'] #The contract looked really weird. Always same motion. 
#Removed GoPro1 because I would get OOM error with 512x512 videos. That would only happen on the last video, because of video_segment init. 
for i in range(len(names)):
    names[i] = names[i] + "_" + video_type + ".mp4"
#names = ['our_planet.mp4']
#names = ["GoPro1.mp4"]
#Disables memory allocation. Makes the code run slower, in theory. 
cp.cuda.set_allocator(None)


max_mem = 95


#Use second GPU
cp.cuda.runtime.setDevice(1)

global_normalize = True

###NOTE I CHANGED BIN_PSD TO NOT TAKE LOW SPATIAL FREQUENCIES IN X OR Y AS A TEST!!!

#hey = PSD(names, 250, 128); hey.average_TS_PSD()

def cuda_mem():
    print(cp.get_default_memory_pool().used_bytes()/(1+cp.get_default_memory_pool().total_bytes()))
cuda_mem()
global_path_pre = os.getcwd() + "/../../video_color_PSD/" 
global_path = global_path_pre + video_type + "/" #/home/david/Github/video_color_PSD/"
#Taken from: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
adobergb2xyz = cp.array([[0.5767309, 0.1855540, 0.1881852],
                            [0.2973769, 0.6273491, 0.0752741],
                            [0.0270343, 0.0706872, 0.9911085]])

srgb2xyz = cp.array([[0.4127, 0.3586, 0.1808],
                     [0.2132, 0.7172, 0.0724],
                     [0.0195, 0.1197, 0.9517]])
                    
#Taken from Colorimetry book, page 305. 
xyz2lms = cp.array([[0.4002,  0.7075, 0.0807],
                        [0.2280, 1.1500, 0.0612],
                        [0,      0,      0.9184]])

#Taken from the Colorimetry book, page 227. 10 degrees 
rgb2lms = cp.array([[0.192325269,  0.749548882,  0.0675726702],
                       [0.0192290085, 0.940908496,  0.113830196],
                       [0,            0.0105107859, 0.991427669]])

#Taken from: http://www.scikit-video.org/stable/examples/io.html
#Reads a YUV video with no conversion

#Takes 2D spectrogram in spacexspace and returns radial freqs and rad power


def create_video_cropped(n_pixels, names):
    for video_name in names:
        clip = moviepy.editor.VideoFileClip(global_path + video_name)
        subclip = crop(clip,width = n_pixels, height = n_pixels, x_center = int(clip.size[0]/2), y_center = (clip.size[1]/2))
        subclip.write_videofile(global_path + video_name.replace(".mp4","") + "_crop" + str(n_pixels) + ".mp4")

def check_memory(maximum, print_statement = True):
    if sys.platform == 'win32':
        return
    total_memory, used_memory, free_memory = map(
    int, os.popen('free -t -m').readlines()[-1].split()[1:])
    if print_statement:
        print("Currently using " + str(used_memory/total_memory*100) + "% of memory")
    if used_memory/total_memory > maximum/100:
        raise Exception("You're using too much memory, stop!")
    
    
    #Note 1: Run same plot but with log spatial frequency
    #Note 2: Run same analysis with different window sizes and see if we have same result 
#This function is obsolete. Use load_video_segment instead. 
#def load_video(filename, n_frames = 10000, path = global_path):
#    check_memory(max_mem)
#    return skvideo.io.vread(path + filename, num_frames = n_frames)[:,:,0:500,:]

#This was an attempt to make load_video_segment faster. It helped me understand moviepy, but I doubt its faster. The subclip function uses get_frame in a for loop anyways.
#Can be used to try new things. 
def load_video_segment_sandbox(video_clip,frames):
    print('hi')
    fps = video_clip.fps
    print(frames)
    subclip = video_clip.subclip(frames[0].get()/fps, frames[1].get()/fps) #This function doesn't return the exact number of frames I need, leading to a shape error later. 
    #Should use a for loop with get_frame instead. 
    
    size = subclip.size
    subclip_crop = crop(subclip, width = freq_length, height = freq_length, x_center = int(size[0]/2), y_center = int(size[1]/2))
    
    video_segment = cp.array(list(subclip_crop.iter_frames()))
    print('hi again')
    return video_segment

def load_video_segment(video_clip, frames):
    fps = video_clip.fps
    check_memory(max_mem, print_statement = False)
    video_segment = []
    first_frame = video_clip.get_frame(10)
    frames_array = cp.linspace(int(frames[0]), int(frames[1]), int(frames[1] - frames[0] + 1))
    #frames_array = cp.array(range(frames[0], frames[1]))
    x_center = int(first_frame.shape[0]/2)
    y_center = int(first_frame.shape[1]/2)
    
    x_length = freq_length
    y_length = freq_length
    
    x_min = int(x_center-x_length/2)
    x_max = int(x_center+x_length/2)
    
    y_min = int(y_center-y_length/2)
    y_max = int(y_center+y_length/2)
    for t in frames_array:
        video_segment.append(video_clip.get_frame(t/fps))
    #print(first_frame.shape, cp.max(first_frame))
    
    #return cp.array(video_segment)[:,x_min:x_max, y_min:y_max,:] #ORIGINAL CX WAS 100x100
    return cp.array(video_segment)[:,x_min:x_max, y_min:y_max,:]
    #return cp.array(video_segment)[:,130:180, 270:320,:]
    #return cp.array(video_segment)


def linearize_adobergb(video_rgb):
    img_float = video_rgb.astype(cp.float32)
    img_float /= 255
    linear_rgb = img_float**2.2
    return linear_rgb

def linearize_srgb(video_rgb):
    linear_rgb = cp.zeros(video_rgb.shape)
    img_float =  video_rgb.astype(cp.float32)
    img_float /= 255
    above_thresh = img_float > 0.04045
    linear_rgb[cp.invert(above_thresh)] = img_float[cp.invert(above_thresh)]/12.92
    linear_rgb[above_thresh] = ((img_float[above_thresh] + 0.055)/1.055)**2.4
    return linear_rgb

class video_segment():
    #Takes as input a video object from the moviepy.editor package. From this video, takes a segment from index_range. 
    #This segment has images with 8bits integer values ranging from 0 to 255 in srgb, and converts it to 0 to 1 floats in lms space.  
    #Index_range is a list with 2 values, which represent the start and stop frames of the video slice. 
    
    
    def __init__(self, video_clip, frames, color_means, color_stds):
        #index = cp.int(cp.linspace(index_range[0], index_range[1] - 1, num = index_range[1] - index_range[0]))
        video = load_video_segment(video_clip, frames)
        video = linearize_srgb(video)
        video = cp.dot(srgb2xyz, cp.swapaxes(video, 3, 2))
        video = cp.dot(xyz2lms, cp.swapaxes(video, 0, 2))
        video = cp.swapaxes(cp.swapaxes(video, 0, 2), 1, 2)
        #video_lms = self.normalize_color(video_lms)
        self.n_colors = video.shape[1]
        for c in range(self.n_colors):
          #  for t in range(video_lms.shape[0]):
            
                video[:,c,:,:] = (video[:,c,:,:] - color_means[c])/color_stds[c]
                #print(cp.mean(video_lms[t,c,:,:]), cp.std(video_lms[t,c,:,:]))
                #for t in range(video_lms.shape[0]):
                #    video_lms[t,c,:,:] = video_lms[t,c,:,:] - cp.mean(video_lms[t,c,:,:])#/cp.std(video_lms[t,c,:,:])
                #print(video_lms.shape)
            
        self.video = video 
    
    
    
    def init_old(self, video_clip, frames, color_means, color_stds):
        #index = cp.int(cp.linspace(index_range[0], index_range[1] - 1, num = index_range[1] - index_range[0]))
        self.video_rgb = load_video_segment(video_clip, frames)
        linear_rgb = self.linearize_srgb()
        xyz = cp.dot(srgb2xyz, cp.swapaxes(linear_rgb, 3, 2))
        lms = cp.dot(xyz2lms, cp.swapaxes(xyz, 0, 2))
        video_lms = cp.swapaxes(cp.swapaxes(lms, 0, 2), 1, 2)
        #video_lms = self.normalize_color(video_lms)
        self.n_colors = self.video_rgb.shape[3]
        for c in range(self.n_colors):
          #  for t in range(video_lms.shape[0]):
            
                video_lms[:,c,:,:] = (video_lms[:,c,:,:] - color_means[c])/color_stds[c]
                #print(cp.mean(video_lms[t,c,:,:]), cp.std(video_lms[t,c,:,:]))
                #for t in range(video_lms.shape[0]):
                #    video_lms[t,c,:,:] = video_lms[t,c,:,:] - cp.mean(video_lms[t,c,:,:])#/cp.std(video_lms[t,c,:,:])
                #print(video_lms.shape)
            
        self.video = video_lms 
        
        

    
    
    
    #Normalize each color channel to have mean = 0 and std = 1. Otherwise, S channel just has more power (?)
    def normalize_color(self, video):
        for c in range(3):
            mean = cp.mean(video[:,c,:,:])
            std = cp.std(video[:,c,:,:])
            video[:,c,:,:] -= mean
            video[:,c,:,:] /= std
            #print(video.shape, mean, std, c)
        return video
    #Does a 3D PSD on self.video. Only takes real freqs
    def make_psd_3D(self):
        real_spatial_freqs = int(self.video.shape[2]/2)
        real_temporal_freqs = int(self.video.shape[0]/2)
        #psd2D_space = psd2D_space[:, 0:real_spatial_freqs, 0:real_spatial_freqs]

        f3D = cp.fft.fftn(self.video, axes = (0,2,3))
        psd3D = abs(f3D)**2
#<<<<<<< Updated upstream
        #f3D_real = f3D[0:real_temporal_freqs, :, 0:real_spatial_freqs, 0:real_spatial_freqs]num
        #Cx = cp.einsum('ijkl,ajkl->iajkl', f3D_real, f3D_real)
        
        #self.f3D = abs(f3D_real)
        self.f3D = f3D
        self.psd3D = psd3D[0:real_temporal_freqs, :, 0:real_spatial_freqs, 0:real_spatial_freqs]
        
    def make_Cx(self):
        real_spatial_freqs = int(self.video.shape[2]/2)
        real_temporal_freqs = int(self.video.shape[0]/2)
        #psd2D_space = psd2D_space[:, 0:real_spatial_freqs, 0:real_spatial_freqs]

        f3D = cp.fft.fftn(self.video, axes = (0,2,3))
    
    def radial_space_freq(self, space_psd):
        size = space_psd.shape[1]
        freqs = cp.fft.fftfreq(size*2)
        freqs_norm = []
        power = []
        for i in range(int(size)):
            for j in range(int(size)):
                freqs_norm.append(cp.sqrt(freqs[i]**2 + freqs[j]**2))
                power.append(space_psd[i, j])
        return freqs_norm, power
    
    
    #Takes as input 1D arrays that represent frequencies (output of radial_space_freq), power and n_bins. Default is 10. 
    def bin_1D_psd_old(self, freqs, power, n_bins):
        freqs_np = cp.array(freqs).get()
        power_np = cp.array(power).get()
        
        freqs_space_bin = scipy.stats.binned_statistic(freqs_np, freqs_np, bins = n_bins)[0]
        power_space_bin = scipy.stats.binned_statistic(freqs_np, power_np, bins = n_bins)[0]
        return cp.asarray(freqs_space_bin), cp.asarray(power_space_bin)
    
    
    #For every temporal frequency, look at the 3D psd for that TF and compute the radial spatial frequencies. Then, bin those radial spatial frequencies in n_bins different bins of the same size. 
    #This returns Cx 
    def make_spatiotemporal_psd(self, n_bins, save_radial_freqs = False, min_freq = 0):
        n_TFs = self.psd3D.shape[0]*2 + 1

        size = self.psd3D.shape[2]
        STpsd = []
        
        #freqs = cp.fft.fftfreq(size*2)[0:int(size)] #This truncates the fft to only take half the freqs. 
        freqs = cp.fft.fftfreq(size*2)
        y_freqs = cp.repeat(cp.repeat(cp.repeat(freqs[cp.newaxis,cp.newaxis,:, cp.newaxis], n_TFs, axis = 0), self.n_colors, axis = 1), freqs.shape[0], axis = 3)
        x_freqs = cp.repeat(cp.repeat(cp.repeat(freqs[cp.newaxis,cp.newaxis,cp.newaxis,:], n_TFs, axis = 0), self.n_colors, axis = 1), freqs.shape[0], axis = 2)
        
        
        freqs_space = cp.sqrt(x_freqs**2 + y_freqs**2)
        if save_radial_freqs:
            self.freqs_space = freqs_space #Remove this line if you run out of memory, its meant to be temporary
        
        #freqs_bin = self.bin_psd(freqs_space[:,:,min_freq:-1,min_freq:-1], self.f3D[:,:,min_freq:-1,min_freq:-1], n_bins)
        #self.freqs_bin = freqs_bin
        
        #Cx = cp.einsum('ijk,ijl->ijkl', freqs_bin, cp.conjugate(freqs_bin)) #from radial freqs
        #Cx = cp.einsum('tcxy,tsxy->tcsxy', self.f3D, cp.conjugate(self.f3D))
        Cx = cp.einsum('tcxy,tsxy->tcsxy', cp.conjugate(self.f3D), self.f3D)
        #power_bin2 = freqs_bin**2
        #power_bin = self.bin_psd(freqs_space[:,:,min_freq:-1,min_freq:-1], self.psd3D[:,:,min_freq:-1,min_freq:-1], n_bins)
        #self.STpsd = cp.log10(power_bin)
        #self.STpsd_test = cp.log10(power_bin2)
        #self.Cx = cp.log10(Cx)
        self.Cx = Cx


class Video():
    def __init__(self, video_name, time_bins, path = global_path, frames = None):
        self.video_name = video_name
        video_full = moviepy.editor.VideoFileClip(path + video_name)
        size = video_full.size
        self.video = video_full# crop(video_full, width = freq_length, height = freq_length, x_center = int(size[0]/2), y_center = int(size[1]/2))
        
        #try:
        #    video_clip = moviepy.editor.VideoFileClip(path + video_name)
        #    self.video_clip = video_clip#.resize(0.1)
        #except FileNotFoundError:
        #    print("Could not find file " + path + video_name)
        if frames is None:
            self.max_frame = int(self.video.duration*self.video.fps)
            self.min_frame = 0
        else:
            self.max_frame = frames[1]
            self.min_frame = frames[0]
        
        self.time_points = cp.arange(self.min_frame, self.max_frame, time_bins)
        self.color_means = cp.array([0,0,0])
        self.color_stds = cp.array([1,1,1])
        if global_normalize:
            self.compute_means()
        print("Hello")
        
        
        
    def compute_means(self):
        filename = "stats.csv"
        df = pd.read_csv(global_path_pre + filename)
        
        if self.video_name in list(df['Video']):
            print("Reading mean and std for each color channel")
            df_row = df[df['Video'] == self.video_name]
            mean_cols = ['L_mean', 'M_mean', 'S_mean']
            std_cols = ['L_std', 'M_std', 'S_std']
            means = []
            stds = []
            for c in range(3):
                means.append(float(df_row[mean_cols[c]]))
                stds.append(float(df_row[std_cols[c]]))
            print(means, stds)
            self.color_means = cp.array(means)
            self.color_stds = cp.array(stds)
        else:
            print("Computing mean and std of each color channel")
            means_sum = 0
            stds_sum = 0
            for t in range(self.time_points.shape[0]-1):
                segment = video_segment(self.video, [self.time_points[t], self.time_points[t+1]], [0,0,0], [1,1,1])
                mean = cp.mean(segment.video, axis = (0,2,3))
                std = cp.std(segment.video, axis = (0,2,3))
                means_sum += mean
                stds_sum += std
                if t%100 == 0:
                    print(self.time_points[t], means_sum/(t + 1), stds_sum/(t + 1))
            print("Done computing mean and std of each color channel")
            self.color_means = means_sum/(self.time_points.shape[0]-1)
            self.color_stds = stds_sum/(self.time_points.shape[0]-1)
            
            row = [self.video_name]
            row.extend(self.color_means.get())
            row.extend(self.color_stds.get())
            
            
            if df['Video'].str.contains(self.video_name).any():
                index_bool = df['Video'] == self.video_name
                index = df.loc[index_bool].index[0] 
                df.loc[index] = row
            else:
                df.loc[len(df)] = row
            df.to_csv(global_path_pre + filename, index=False)
        
        
class PSD():
    #Frames is an array with 2 values: [frame_min, frame_max]
    def __init__(self, video_names, time_bins, n_spatial_bins, frames = None, path = global_path):
        
        self.n_spatial_bins = n_spatial_bins
        self.time_bins = time_bins
        
        videos = []
        for video_name in video_names:
            video = Video(video_name, time_bins, frames = frames, path = path)
            videos.append(video)
        self.videos = videos
        
        

        
   
        
    #time_bins = how long should each segment be. n_spatial_bins = we need to bin radial frequencies, how many bins do you want in total.
    def average_TS_PSD(self):
        current_time = time.time()
        
        new = True
        all_PSD = []
        total_segments = 0
        for video in self.videos:
            print(video.video_name)
            for t in range(video.time_points.shape[0]-1):
                total_segments += 1
                #print("Start:", time.time() - current_time)
                check_memory(max_mem, print_statement = False)
                segment = video_segment(video.video, [video.time_points[t], video.time_points[t+1]], video.color_means, video.color_stds)
                #print("Made segment:", time.time() - current_time)
                segment.make_psd_3D()
                #print("Make_psd_3D:", time.time() - current_time)
                segment.make_spatiotemporal_psd(self.n_spatial_bins, True) #This is where the power values go through log10
                #print("Spatiotemporal_psd", time.time() - current_time)
                
                #psd = segment.STpsd_test
                #psd3d = segment.psd3D
                
                Cx_seg = segment.Cx
                #self.last_segment = segment
                #all_PSD.append(segment.STpsd)
                if new:
                    #psd_sum = psd
                    #psd3d_sum = psd3d
                    new = False
                    Cx_sum = Cx_seg
                else: 
                    if not cp.max(abs(Cx_seg)) == cp.inf and not cp.any(cp.isnan(Cx_seg)):
                        Cx_sum += Cx_seg
                        #psd_sum += psd
                        #psd3d_sum += psd3d
                    else:
                        
                        print("Removed segment number:", str(t))
                        self.problem = Cx_seg
                        self.problem2 = segment
                if t%100 == 0:
                    print(video.time_points[t])
        #self.PSD = (psd_sum/self.time_points.shape[0]).get()
        #self.PSD3D = psd3d_sum/self.time_points.shape[0]
        self.all_PSD = cp.array(all_PSD).get()
        
        Cx = (Cx_sum/total_segments)
        self.Cx = Cx.get()
        save_params(self,"intermediate")
        
        
        Cx_bin = bin_Cx(Cx,Cx_n_bins)
        self.Cx_bin = Cx_bin.get()
        self.Cx_eigendecomp(self.Cx_bin)
        print(Cx.shape, 'Cx shape')

        print("Finish:", current_time - time.time())
        cuda_mem()
    
    #Need self.Cx_bin to exist and be on cpu
    def Cx_eigendecomp(self):
        Cx = self.Cx_bin
        n_space_freqs = Cx.shape[0]
        n_time_freqs = Cx.shape[1]
        n_colors = Cx.shape[2]
        
        if cp.array(Cx.shape).shape[0] < 5:
            eigvals = numpy.zeros([n_space_freqs, n_time_freqs, n_colors])
            eigvects = numpy.zeros([n_space_freqs, n_time_freqs, n_colors, n_colors])
            #Solve generalized eigenvalue problem of 
            
            for space_freq in range(n_space_freqs):
                for time_freq in range(n_time_freqs):
                    eig = scipy.linalg.eigh(Cx[space_freq, time_freq, :, :])
                    eigvals[space_freq,time_freq,:] = eig[0]
                    eigvects[space_freq, time_freq, :,:] = eig[1]
        else:
            eigvals = numpy.zeros([Cx.shape[0], n_colors, Cx.shape[-1], Cx.shape[-1]])
            eigvects = numpy.zeros([Cx.shape[0], n_colors, n_colors, Cx.shape[-1], Cx.shape[-1]])
            for space_freq1 in range(Cx.shape[-1]):
                for space_freq2 in range(Cx.shape[-1]):
                    for time_freq in range(Cx.shape[0]):
                        eig = scipy.linalg.eigh(Cx[time_freq, :, :, space_freq1, space_freq2])
                        eigvals[time_freq,:,space_freq1, space_freq2] = eig[0]
                        eigvects[time_freq, :,:, space_freq1, space_freq2] = eig[1]
                
                
        self.Cx_eigvals = eigvals
        self.Cx_eigvects = eigvects
        
        
    def import_psd(self, filename, path = global_path):
        self.PSD = cp.load(path + filename)
        
    def save_psd(self, filename, path = global_path):
        cp.save(path + filename, self.PSD)
    
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
    

        
def bin_psd(power, n_bins):
    n_TFs = power.shape[0]
    size = power.shape[-1]
    freqs = cp.fft.fftfreq(size)
    n_colors = 3
    y_freqs = cp.repeat(cp.repeat(cp.repeat(freqs[cp.newaxis,cp.newaxis,:, cp.newaxis], n_TFs, axis = 0), n_colors, axis = 1), freqs.shape[0], axis = 3)
    x_freqs = cp.repeat(cp.repeat(cp.repeat(freqs[cp.newaxis,cp.newaxis,cp.newaxis,:], n_TFs, axis = 0), n_colors, axis = 1), freqs.shape[0], axis = 2)
    
    
    freqs_space = cp.sqrt(x_freqs**2 + y_freqs**2)
    
    bins = numpy.linspace(cp.min(freqs_space), cp.max(freqs_space), n_bins)
    digitized = cp.digitize(freqs_space, bins)

    #bin_means = [power[digitized == i].reshape([self.psd3D.shape[0],self.n_colors, -1]).mean(axis=2) for i in range(1, len(bins)-1)] #I ran tests and this should be the right order
    bin_means2 = []
    for i in range(1, len(bins) - 1):
        mean = power[digitized == i].reshape([n_TFs,n_colors, -1]).mean(axis=2)
        if not cp.isnan(mean[0,0]):
            bin_means2.append(mean)
    return cp.array(bin_means2)


def bin_Cx(Cx, n_bins):
    n_TFs = Cx.shape[0]
    size = Cx.shape[-1]
    freqs = cp.fft.fftfreq(size)
    n_colors = 3
    y_freqs = cp.repeat(cp.repeat(cp.repeat(cp.repeat(freqs[cp.newaxis,cp.newaxis, cp.newaxis, :, cp.newaxis], n_TFs, axis = 0), n_colors, axis = 1), n_colors, axis = 2), freqs.shape[0], axis = 4)
    x_freqs = cp.repeat(cp.repeat(cp.repeat(cp.repeat(freqs[cp.newaxis,cp.newaxis,cp.newaxis, cp.newaxis, :], n_TFs, axis = 0), n_colors, axis = 1), n_colors, axis = 2), freqs.shape[0], axis = 3)
    
    
    freqs_space = cp.sqrt(x_freqs**2 + y_freqs**2)
    
    bins = numpy.linspace(cp.min(freqs_space), cp.max(x_freqs), n_bins)
    #bins = numpy.linspace(cp.min(freqs_space), cp.max(freqs_space), n_bins)
    
    digitized = cp.digitize(freqs_space, bins)
    angle = cp.arctan((x_freqs + 0.0001)/(y_freqs + 0.0001))
    print(cp.max(angle), cp.min(angle))
    #bin_means = [power[digitized == i].reshape([self.psd3D.shape[0],self.n_colors, -1]).mean(axis=2) for i in range(1, len(bins)-1)] #I ran tests and this should be the right order
    bin_means = []
    for i in range(1, len(bins) - 1):
        #means = Cx[digitized == i].reshape([n_TFs,n_colors,n_colors, -1])
        
        #Only take angles away from 0 or pi/2. Right thing to do!
        #means = Cx[cp.logical_and(digitized == i, cp.logical_and(abs(angle) > 0.1, abs(angle)< cp.pi/2 - 0.1))].reshape([n_TFs,n_colors,n_colors, -1])
        means = Cx[cp.logical_and(digitized == i, cp.logical_and(abs(angle) > cp.pi/4 -0.1, abs(angle) < cp.pi/4 +0.1))].reshape([n_TFs,n_colors,n_colors, -1])
        mean = means.mean(axis=3)
        if not cp.isnan(mean[0,0,0]):
            bin_means.append(mean.get())
    return cp.array(bin_means)

def plot_psd(to_plot, axis = None, title = "", divergent = False, v = None):
    colorbar = False
    if axis is None:
        fig, axis = plt.subplots(1,1)
        colorbar = True

    if v is None:
        vmax = numpy.max(cp.array([abs(numpy.min(to_plot)), numpy.max(to_plot)]))
    else:
        vmax = v
    if divergent:
        color_map = 'PiYG'
        vmin = vmax*-1
    else:
        color_map = 'YlOrRd'
        vmin = 0
    
    s = axis.imshow(to_plot, origin = 'lower', cmap = color_map, vmin = vmin, vmax = vmax)
    #axis.contour(to_plot, origin = 'lower', color = 'black', alpha = 1, linewidths = 3)

    if colorbar:
        cbar = fig.colorbar(s)
        cbar.ax.tick_params(labelsize=50)
        axis.set_xlabel("Temporal frequency", size = 50)
        axis.set_ylabel("Spatial frequency", size = 50)
        axis.set_title(title, size = 50)
    
    axis.set_xticks([])
    axis.set_yticks([])

def plot_psd_cov(to_plot):
    #last two dimensions define subplot size
    n_rows = to_plot.shape[2]
    n_cols = to_plot.shape[3]
    
    if n_cols != n_rows:
        raise Exception("Last two dimensions are not the same")
    fig, ax = plt.subplots(n_rows,n_cols, constrained_layout=True)
    #plt.get_current_fig_manager().window.showMaximized()
    
    
    for j in range(n_cols):
        for i in range(n_rows):
            if i <= j:
                plot_psd(to_plot[:,:,i,j], axis = ax[i,j])
            else:
                ax[i,j].set_visible(False)
    fig.tight_layout()

def plot_eig_diff(to_plot):
    #last two dimensions define subplot size
    n_eig = to_plot.shape[2]
    
    fig, ax = plt.subplots(n_eig,n_eig, constrained_layout=True)
    
    
    for i in range(n_eig):
        for j in range(n_eig):
            if i == j:
                plot_psd(to_plot[:,:,i], axis = ax[i,j])
            elif i > j:
                plot_psd(to_plot[:,:,j] - to_plot[:,:,i], axis = ax[i,j])
            else:   
                ax[i,j].set_visible(False)
    fig.tight_layout()

def save_params(params, name):
    attributes = ['Cx', 'Cx_bin', 'Cx_eigvals', 'Cx_eigvects']
    for attribute in attributes:
        if not hasattr(params, attribute):
            setattr(params, attribute, None)
    
    this_dict = {'Cx': params.Cx, 'Cx_bin': params.Cx_bin, 'Cx_eigvals':params.Cx_eigvals, 'Cx_eigvects': params.Cx_eigvects}
    with open(global_path_pre + '/../' + name + '.pkl', 'wb') as outp:
        pickle.dump(this_dict, outp)
        


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
#hey = PSD("Nature1_lowres.mp4", 250, 100); hey.average_TS_PSD()


#hey.plot_log_spatial_psd(hey.PSD3D, 10)
