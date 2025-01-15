# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:13:14 2024

@author: David
"""

import os
import numpy as np
import pickle
import scipy
import scipy.optimize as opt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.signal as ssig

from scipy.interpolate import RectBivariateSpline, BivariateSpline

#That looks really nice: 
#test = grid_models([0.25,0.5,1,2,4], [1,2,3,4,5,6])
#grid_values(test, command = '[0].powers[2]')

P_total = 1

def soft_bandpass(lo, hi, freqs, stiffness=10):
    if lo <= 0:
        return scipy.special.expit(stiffness * (hi - freqs))
    else:
        return scipy.special.expit(stiffness * (freqs - lo)) * scipy.special.expit(stiffness * (hi - freqs))

def import_C():
    with open(os.getcwd() + "/../../Cx.pkl", 'rb') as f:
        x = pickle.load(f)
    return x

def pad_and_reflect(fil, N, padval=0):
    """
    Zero-pad to length N//2 and reflect about origin to make length N. 
    """
    expanded = np.pad(fil, (0, N//2 + 1 - len(fil)), constant_values=(0, padval))
    return np.concatenate([expanded[::-1], expanded[1:]])

def extrap_and_reflect(fil, N, return_log=False):
    """
    Linearly extrapolate the log filter to better approximate tails and reflect 
    about origin to make length N.
    """
    extrap_fun = interp1d(range(0, len(fil)), np.maximum(-32, np.log(fil)), fill_value='extrapolate')
    expanded = extrap_fun(range(0, N//2 + 1))
    if not return_log:
        expanded = np.exp(expanded)
    return np.concatenate([expanded[::-1], expanded[1:]])

class covariance():
    def __init__(self):
        C_dict = import_C()
        
        self.n_time_freqs = C_dict['Cx_eigvals'].shape[1]//2
        self.n_space_freqs = C_dict['Cx_eigvals'].shape[0]
        Cx_pre = C_dict['Cx_eigvals'][:,0:self.n_time_freqs,:]
        Cx_pre = np.flip(Cx_pre, axis = 2)
        U = C_dict['Cx_eigvects'][:,0:self.n_time_freqs,:]
        U = np.flip(U, axis = 3)
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                for c in range(U.shape[3]):
                    if U[i,j,0,c] < 0:
                        U[i,j,:,c] *= -1
        #Cx = scipy.ndimage.gaussian_filter(Cx_pre, sigma = (2,0,0), order = 0)
        self.Cx = scipy.ndimage.median_filter(Cx_pre, size = 5, axes = 0)
        self.n_channels = self.Cx.shape[2]
        self.eigvects = U
        
    def interpolate(self, k, o, i, scaling):
        func = RectBivariateSpline(np.linspace(0, 2*np.pi, self.Cx.shape[0]), np.linspace(0, 2*np.pi, self.Cx.shape[1]), np.log10(self.Cx[:,:,i]), s = 0)
        return 10**func(k,o)/(10**scaling) #November 7th, 202
#return 10**func(k,o)/1000000

C = covariance()

class model():
    def __init__(self, fixed_freq, n_freqs, klims, P, fixed_type, sigout, scaling):
        self.scaling = scaling
        self.sigout = sigout
        self.fixed_freq = fixed_freq
        self.freqs = np.linspace(0,2*np.pi, n_freqs)
        self.P = P
        self.fixed_type = fixed_type
        self.n_channels = C.n_channels
        self.power_freqs = []
        power_constraint = {'type': 'ineq', 'fun': self.excess_power, 'args': (self.freqs, klims)}
        eps_fun = self.create_information_eps()
        res = opt.minimize(eps_fun, np.array([-10,-10,-10]), bounds=[(-16, np.inf)], constraints=[power_constraint])
        if res.success:
            #print("Excess power: ", excess_power(res.x, fixed_freq, indices, klims, P, fixed_type))
            self.log_nus = res.x
        else:
            print("Optimizer failed to converge!")
            self.log_nus = None
        
        
    def create_information_eps(self, return_sum = True):
        def information_eps(log_eps):
            infos = []
            for i in range(C.n_channels):
                eps_eigenchannel = 10**log_eps[i]
                if self.fixed_type == 'spatial':
                    #ktilde = 1/Cx[int(fixed_freq),:, i]
                    ktilde = 1/C.interpolate(self.fixed_freq, self.freqs, i, self.scaling)
                elif self.fixed_type == 'temporal':
                    #ktilde = 1/Cx[:,int(fixed_freq), i]
                    ktilde = 1/C.interpolate(self.freqs, self.fixed_freq, i, self.scaling)
                
                
                power = self.filter_power(eps_eigenchannel, self.freqs, None, i)
                
                ktilde = pad_and_reflect(ktilde[:,0],ktilde.shape[0]*2 - 2)
                #numer = np.maximum(0, 1/(2 * (ktilde + 1e-32)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * self.sigout**2)) - 1) - 1) + 1
                #denom = np.maximum(0, 1/(2 * (ktilde + 1)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * self.sigout**2)) + 1) - 1) + 1
                
                numer = power + self.sigout**2
                denom = (ktilde/(1+ktilde)*power + self.sigout**2)
                
                self.info_numer = numer
                self.info_denom = denom
                
                info = np.sum(np.log(numer) - np.log(denom))
                infos.append(info)
            if return_sum:
                return np.sum(infos)*-1
            else:
                return infos 
        return information_eps

    def filter_k(self, eps, channel, k_lims=None):
        def v_opt(indices):
            if self.fixed_type == 'spatial':
                inter = np.clip(C.interpolate(self.fixed_freq,indices, channel, self.scaling), 1e-32, np.inf).T
                ktilde = 1/inter
            elif self.fixed_type == 'temporal':
                inter = np.clip(C.interpolate(indices, self.fixed_freq, channel, self.scaling), 1e-32, np.inf)
                ktilde = 1/inter
            ktilde = pad_and_reflect(ktilde[:,0],ktilde.shape[0]*2 - 2)
            sqrt_piece = np.sqrt(1 + (4/(eps*self.sigout**2)) * ktilde)
            
            v2 = 0.5 * (sqrt_piece + 1) / (1 + ktilde) - 1
            v2 = np.sqrt(np.maximum(v2, 0) * self.sigout**2)
            if k_lims:
                unit_cell_k = 0# soft_bandpass(k_lims[0], k_lims[1], k)
                v2 *= unit_cell_k
            return v2
        return v_opt
    
    def filter(self, nu, k_lims=None):
        def v_opt(k, o, i):
            CC = np.minimum(C.interpolate(k, o, i, self.scaling), 1e32)
            sqrt_piece = np.sqrt(CC**2 + (4/nu) * (1/self.sigout**2) * CC)
            v2 = 0.5 * (sqrt_piece + CC) / (1 + CC) - 1
            v2 = np.sqrt(np.maximum(v2, 0) * self.sigout**2)
            if k_lims:
                unit_cell_k = 0#soft_bandpass(k_lims[0], k_lims[1], k)
                v2 *= unit_cell_k
    
            return v2 #- np.min(v2)
        return v_opt
    
    def filter_power(self, eps, indices, klims, channel, sum_power = True):
        vfun = self.filter_k(eps, channel, klims)
        v2 = vfun(indices)
        if self.fixed_type == 'spatial':
            ktilde = 1/C.interpolate(self.fixed_freq, indices, channel, self.scaling)
        elif self.fixed_type == 'temporal':
            ktilde = 1/C.interpolate(indices, self.fixed_freq, channel, self.scaling)
        
        ktilde = pad_and_reflect(ktilde[:,0],ktilde.shape[0]*2 - 2)
    
        dk = 2*np.pi/len(indices)
    
    
        freqs = pad_and_reflect(self.freqs, len(self.freqs)*2 - 2)
        self.v2 = v2
        power_freqs = v2**2 * np.abs(freqs) * (1/ktilde + 1) * dk
        if sum_power: 
            return np.sum(power_freqs)/(2*np.pi)
            
        else:
            power_freqs = power_freqs/(2*np.pi)
            self.power_freqs.append(power_freqs)
            #return power_freqs
        #return np.sum(v2**2 * np.abs(freqs) * (1/ktilde + 1) * dk)/(2 * np.pi)
        #John: Needs to make sure I do the conversion from integral to sum correctly
        #Need to make srue dk is the correct value. dk = 2*pi/L
        #By convention, frequencies discrete are -pi to pi. My frequencies are from 0 to L. 

    def excess_power(self, log_eps, indices, klims):
        total_power = 0
        eig_powers = []
        for i in range(C.n_channels):
            eig_power = self.filter_power(10**log_eps[i], indices, klims, i)
            total_power += eig_power
            eig_powers.append(eig_power)
        self.powers = eig_powers
        return self.P - total_power
    
    
    def get_v(self):
        # make a double exponential smoothing filter
        fz = np.arange(-.2,.2, 0.00005) #0.0015 for 1000 and 0.0001 for 10000
        ff = np.exp(-20 * np.abs(fz))
        ff /= np.sum(ff)
        
        vspace_omega = []; vv_all = []; vf_all = []; vvf_all = []; log_va_all = []
        for i in range(self.n_channels):
            self.filter_power(10**self.log_nus[i], self.freqs, None, i, sum_power = False)
            vv = self.filter_k(10**self.log_nus[i], i)(self.freqs)
            vv = vv[vv.shape[0]//2:]
            vf = np.convolve(vv, ff, mode = 'same')
            #FOR THE TEMPORAL CASE, THIS CONVOLUTION IS NOT ENOUGH AT ALL 
            
            if self.fixed_type == 'temporal':
                vvf = pad_and_reflect(vf, vf.shape[0]*2)
                log_va = 0
                vspace = np.real(scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(vvf))))
            elif self.fixed_type == 'spatial':
                vvf = extrap_and_reflect(vf, vv.shape[0]*2, return_log=True)
                log_va = np.conj(ssig.hilbert(vvf))
                vspace = np.real(scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(np.exp(log_va)))))
            vspace_omega.append(vspace); vv_all.append(vv); vf_all.append(vf); vvf_all.append(vvf); log_va_all.append(log_va)
        vspace_omega = np.array(vspace_omega)
        vspace_omega /= np.linalg.norm(vspace_omega)
        self.vspace = vspace_omega; self.vv = vv_all; self.vf = np.array(vf_all); self.vvf = vvf_all; self.log_va = log_va_all; self.ff = ff
        
    def lms_RFs(self):
        lms_RFs = np.zeros(self.vf.shape)
        vf = self.vf
        n_channels = self.n_channels

        for eig_c in range(n_channels):
            lms_RFs += (np.sqrt(vf[eig_c,:,np.newaxis])*C.eigvects[0,0,:,eig_c]).T
        vspace_LMS = []
        for color in range(n_channels):
            vvf = pad_and_reflect(lms_RFs[color,:], lms_RFs.shape[1]*2)
            vspace = np.real(scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(vvf))))
            vspace_LMS.append(vspace)
        vspace_LMS /= np.linalg.norm(np.array(vspace_LMS))
        self.vspace_LMS = vspace_LMS
            #line, = ax[color].plot(vspace, color = colors[omega_index], label = str(('{:.3f}').format(omega)))
            #ax[color].set_title(lms_titles[color], size = 30)
            #center = int(vspace.shape[0]/2)
            #plot_range = 50
            #ax[color].set_xlim(center - plot_range, center + plot_range)
        #lines.append(line)
    #ax[2].legend(handles=lines, title = "Temporal frequency", fontsize = 12)
        
       
            
def train_filters(n_fixed_freq, n_freqs, fixed_type, sigout, scaling):
    fixed_freqs = np.linspace(0,2*np.pi, n_fixed_freq)
    models = []
    for fixed_freq in fixed_freqs:
        #print(fixed_freq)
        mod = model(fixed_freq, n_freqs, None, P_total, fixed_type, sigout, scaling)
        mod.get_v()
        mod.lms_RFs()
        models.append(mod)
    
    return models

def plot_filters(models, plot_range = 50, lms = False):
    fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'yellow', 'pink', 'darkred', 'olive']
    if not lms:
        channel_labels = ['L+M+S', 'L+M-S', 'L-M']
    else:
        channel_labels = ['L', 'M', 'S']
    lines = []
    x_length = models[0].vspace.shape[1]
    center = x_length//2
    fixed_type = models[0].fixed_type
    
    if fixed_type == 'temporal':
        label_title = "Temporal frequency"
        x_label = "Spatial filter v(z) for "
    elif fixed_type == 'spatial':
        label_title = 'Spatial frequency'
        x_label = 'Temporal filter v(z) for '
    ymin = 0
    ymax = 0
    for omega in range(len(models)):
        omega = 0
        model = models[omega]
        if lms:
            vspace = model.vspace_LMS
        else:
            vspace = model.vspace
        cur_min = np.min(vspace)
        cur_max = np.max(vspace)
        if cur_min < ymin:
            ymin = cur_min
        if cur_max > ymax:
            ymax = cur_max
        
        for i in range(C.n_channels):
            if np.max(vspace[i,:]) == 0:
                print("Zero vspace for kf: ", omega, i)
            if model.fixed_type == 'temporal':
                x_values = np.linspace(-plot_range, plot_range, plot_range*2)
                y_values = vspace[i,center-plot_range:center+plot_range]
            if model.fixed_type == 'spatial':
                x_values = np.linspace(-plot_range, 0, plot_range)
                y_values = vspace[i,center-plot_range:center]
            line, = ax[i].plot(x_values, y_values, color = colors[omega], label = str(('{:.3f}').format(model.fixed_freq)))
            
            ax[i].set_xlabel(r"$z$")
            ax[i].set_title(x_label + channel_labels[i], size = 30);
        lines.append(line)   
    ax[C.n_channels-1].legend(handles=lines, title = label_title, fontsize = 25, title_fontsize=20)
    for axis in ax:
        axis.set_ylim([ymin, ymax])
#THERE IS A BUG WITH ARGMAX PLEASE FIX IT BEFORE USE!!!! THERE IS A BUG WITH ARGMAX PLEASE FIX IT BEFORE USE!!!! THERE IS A BUG WITH ARGMAX PLEASE FIX IT BEFORE USE!!!! THERE IS A BUG WITH ARGMAX PLEASE FIX IT BEFORE USE!!!!
def plot_spacetime(models):
    fig, ax = plt.subplots(1,3, figsize=(16,8))
    RFs = []
    scaling = models[0].scaling
    n_space_freqs = models[0].freqs.shape[0]
    k_indices = np.linspace(0, 2*np.pi, n_space_freqs)
    n_time_freqs = 1000
    w_indices = np.linspace(0, 2*np.pi, n_time_freqs)
    channel_labels = ['L+M+S', 'L+M-S', 'L-M']
    percentile_values = [99, 99.5, 99.9]
    if models[0].fixed_type != 'temporal':
        Exception("Models must be spatial filters!")
    for omega in range(len(models)):
        model_space = models[omega]
        vf = model_space.vf
        argmax = 2*np.pi*(np.argmax(vf)%vf.shape[1])/vf.shape[1]
        #print(argmax, np.argmax(vf), vf.shape)
        model_time = model(argmax, n_time_freqs, None, 1, 'temporal', model_space.P, scaling)
        model_time.get_v()
        vt = model_time.vf
        RF_omega = []
        for i in range(model_space.n_channels):
            RF_color_pre = np.outer(vf[i,:], vt[i,:])
            RF_color = np.roll(RF_color_pre, int(omega/(2*np.pi)*model_space.freqs.shape[0]), axis = 1)
            RF_omega.append(RF_color)
        RFs.append(RF_omega)
    RFs = np.array(RFs)
    for i in range(model_space.n_channels):
        ax[i].imshow(np.log(C.interpolate(k_indices,w_indices,i,scaling = 0)), origin = 'lower', cmap = 'YlOrRd', vmin = 0, vmax = np.max(np.log(C.Cx))) #C_INTERPOLATE HERE!!!!!
        for omega in range(len(models)):
            contour_values = []
            for percentile in percentile_values:
                contour_values.append(np.percentile(RFs[omega,i,:,:], percentile) + percentile*10e-32)
            ax[i].contour(RFs[omega, i,:,:], contour_values, origin = 'lower', alpha = 0.5)
        ax[i].set_title(channel_labels[i])
        ax[i].set_xlabel("Temporal frequency", size = 12)
        ax[i].set_ylabel("Spatial frequency", size = 12)

            
def grid_models(sigs, scales):
    models_all = []
    for i in range(len(sigs)):
        models_sig = []
        for j in range(len(scales)):
            models = train_filters(4, 10000, 'temporal', sigs[i], scales[j])
            models_sig.append(models)
            print(i, j)
        models_all.append(models_sig)
    return models_all

def grid_values(models_all, sigs = [0.25,0.5,1], command = '[0].powers[2]'):
    plt.figure()
    n_sigs = len(models_all)
    n_scales = len(models_all[0])
    grid = np.zeros([n_sigs, n_scales])
    for i in range(n_sigs):
        #for j in list(reversed(range(n_scales))):
         for j in range(n_scales):
             command_post = 'models_all[' + str(i) + '][' + str(j) + ']' + command
             grid[i,j] = eval(command_post)
    grid_flip = np.flip(np.array(grid), axis = 1)
    plt.imshow(grid_flip, origin='lower')
    plt.ylabel("Output noise", size = 30)
    plt.yticks(np.arange(n_sigs), sigs)
    plt.xlabel("Input signal (log10)", size = 30)
    plt.xticks([])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)

    plt.title("Power in L-M eigenchannel at the lowest temporal frequency", size = 30)
    return grid

