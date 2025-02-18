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

#That looks really nice. Can't have output noise more than 1 after power bug fix 
#test = grid_models([0.25,0.5,1], [1,2,3,4,5,6])
#grid_values(test, command = '[0].powers[2]')


#test = grid_models([0.06125,0.125,0.25,0.5,1], [1,2,3,4,5,6]) #To get clean transition through 0

P_total = 1

def soft_bandpass(lo, hi, freqs, stiffness=10):
    if lo <= 0:
        return scipy.special.expit(stiffness * (hi - freqs))
    else:
        return scipy.special.expit(stiffness * (freqs - lo)) * scipy.special.expit(stiffness * (hi - freqs))

def import_C(name):
    with open(os.getcwd() + "/../../" + name + ".pkl", 'rb') as f:
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
    def __init__(self, name):
        C_dict = import_C(name)
        
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
        return 10**func(k,o)/(10**scaling) #November 7th, 2024
    # i is the eigenchannel. Returns eigenvectors number i for different spatial and temporal frequencies. 
    def eigvects_interpolate(self, k, o, i):
        func_L = RectBivariateSpline(np.linspace(0, 2*np.pi, self.eigvects.shape[0]), np.linspace(0, 2*np.pi, self.eigvects.shape[1]), self.eigvects[:,:,0,i], s = 0)
        func_M = RectBivariateSpline(np.linspace(0, 2*np.pi, self.eigvects.shape[0]), np.linspace(0, 2*np.pi, self.eigvects.shape[1]), self.eigvects[:,:,1,i], s = 0)
        func_S = RectBivariateSpline(np.linspace(0, 2*np.pi, self.eigvects.shape[0]), np.linspace(0, 2*np.pi, self.eigvects.shape[1]), self.eigvects[:,:,2,i], s = 0)
        
        #Returns lists of lists but there's no simple way around that if I want the function to generalize for multiple spatial and temporal frequencies.
        return np.array([func_L(k,o), func_M(k,o), func_S(k,o)]) 
    
    
#return 10**func(k,o)/1000000

C = covariance("Cx_256x256")
#C.Cx = C.Cx[0:65,:,:]

#C.eigvects[:,:,:,0] *= -1
#C.eigvects[:,:,:,1] *= -1
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
        self.n_freqs = n_freqs
        
        if fixed_type == 'temporal':
            self.fixed_index = int(fixed_freq*C.n_time_freqs/(2*np.pi))
        if fixed_type == 'spatial':
            self.fixed_index = int(fixed_freq*C.n_space_freqs/(2*np.pi))
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
        fz = np.linspace(-.2,.2, int(self.n_freqs/20)) #0.0015 for 1000 and 0.0001 for 10000
        #fz = np.arange(-.2,.2, 0.0001)
        ff = np.exp(-20 * np.abs(fz))
        ff /= np.sum(ff)
        
        vspace_omega = []; vv_all = []; vf_all = []; vvf_all = []; log_va_all = []
        if self.log_nus is not None:
            for i in range(self.n_channels):
                self.filter_power(10**self.log_nus[i], self.freqs, None, i, sum_power = False)
                vv = self.filter_k(10**self.log_nus[i], i)(self.freqs)
                vv = vv[vv.shape[0]//2:]
                vf = np.convolve(vv, ff, mode = 'same')
                #FOR THE TEMPORAL CASE, THIS CONVOLUTION IS NOT ENOUGH AT ALL 
                #print("hi")
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
            self.vspace = vspace_omega; self.vv = np.array(vv_all); self.vf = np.array(vf_all); self.vvf = vvf_all; self.log_va = log_va_all; self.ff = ff
            #print("hi again")
            return True
        else:
            return False
        
    def lms_RFs(self):
        vv_lms = np.zeros(self.vf.shape)
        vf_lms = np.zeros(self.vf.shape)
        vv = self.vv
        vf = self.vf
        n_channels = self.n_channels
        
        
        for eig_c in range(n_channels):
            eigvects_interpolate = C.eigvects_interpolate(self.freqs,self.fixed_index,eig_c)[:,:,0]
            vv_expand = np.repeat(vv[eig_c,:,np.newaxis],3,1).T
            vf_expand = np.repeat(vf[eig_c,:,np.newaxis],3,1).T

            vv_lms += (vv_expand*eigvects_interpolate)
            vf_lms += (vf_expand*eigvects_interpolate)
            #THERE SHOULD BE NO SQRT HERE!!! THERE IS ALREADY A SQRT IN FILTER_K
            #vv_lms += (vv[eig_c,:,np.newaxis]*C.eigvects[0,self.fixed_index,:,eig_c]).T
            #vf_lms += (vf[eig_c,:,np.newaxis]*C.eigvects[0,self.fixed_index,:,eig_c]).T
        #    vv_lms 
        vspace_LMS = []
        self.vf_lms = vf_lms
        self.vv_lms = vv_lms
        for color in range(n_channels):
            vvf = pad_and_reflect(vf_lms[color,:], vf_lms.shape[1]*2)
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
    fixed_freqs = np.linspace(0.1,2*np.pi, n_fixed_freq)
    #fixed_freq = np.linspace(0, np.pi/4, 8)
    models = []
    for fixed_freq in fixed_freqs:
        #print(fixed_freq)
        mod = model(fixed_freq, n_freqs, None, P_total, fixed_type, sigout, scaling)
        
        success = mod.get_v()
        if success:
            mod.lms_RFs()
            models.append(mod)
        
    return models

def plot_filters(models, ax = None, plot_range = 50, lms = False, omegas = 'all'):
    if ax is None:
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
    if omegas == 'all':
        omegas = range(len(models))
    elif type(omegas) == int:
        omegas = [omegas]
        
    for omega in omegas:
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
        max_index = np.argmax(vf)%model_space.n_freqs
        argmax = 2*np.pi*max_index/model_space.n_freqs
        #print(argmax, np.argmax(vf), vf.shape)
        model_time = model(argmax, n_time_freqs, None, 1, 'spatial', model_space.P, scaling)
        model_time.get_v()
        vt = model_time.vf
        RF_omega = []
        for i in range(model_space.n_channels):
            RF_color_pre = np.outer(vf[i,:], vt[i,:])
            plt.figure()
            plt.imshow(RF_color_pre)
            #print(RF_color_pre.shape)
            RF_color = np.roll(RF_color_pre, int(model_space.fixed_freq*n_time_freqs/(2*np.pi)), axis = 1)
            #RF_color = np.roll(RF_color, int(argmax*model_space.n_freqs/(2*np.pi)))
            #RF_color = RF_color_pre
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

            
def grid_models(sigs, scales, n_freqs = 10000):
    models_all = []
    for i in range(len(sigs)):
        models_sig = []
        for j in range(len(scales)):
            models = train_filters(4, n_freqs, 'temporal', sigs[i], scales[j])
            models_sig.append(models)
            print(i, j)
        models_all.append(models_sig)
    return models_all

#Plot power in L-M channel at lowest spatial frequency for diff input and output noise levels 
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


#Frequency plot for specific temporal frequency, input and output noise levels 
def plot_freqs(models, omegas, log = False, ax = None, lms = False, axis_label = True):
    model = models[omegas]
    if ax is None:
        fig, ax = plt.subplots(1,1)
    lines = []
    
    if lms:
        labels = ['L', 'M', 'S']
        colors = ['red', 'green', 'blue']
        vf = model.vv_lms
    else:
        labels = ['L+M+S', 'L+M-S', 'L-M']
        colors = ['orangered', 'darkorange', 'gold']
        vf = model.vv
    for c in range(model.n_channels):
        if log:
            line, = ax.plot(model.freqs, np.log10(vf[c,:]*np.conjugate(vf[c,:])), color = colors[c], label = labels[c])
        else:
            line, = ax.plot(model.freqs, vf[c,:], color = colors[c], label = labels[c])
            ax.axhline(0)
        #ax.axvline(2.5)
        lines.append(line)
        #line, = ax[i].plot(x_values, y_values, color = colors[omega], label = str(('{:.3f}').format(model.fixed_freq)))
    if axis_label:
        ax.legend(handles=lines)
        ax.set_xlabel("Frequency", size = 30)
        ax.set_ylabel("Power", size = 30)

#My input is a function and a set of models at diff params
def grid_plots(function, models_all, omega, lms, log, sharey = True):
    n_sigs = len(models_all)
    n_scales = len(models_all[0])
    fixed_freq = models_all[0][0][omega].fixed_freq
    fig, axes = plt.subplots(n_sigs, n_scales, sharey = sharey)
    for sig in range(n_sigs):
        for scale in range(n_scales):
            sig_rev = np.abs(sig-n_sigs+1)
            scale_rev = np.abs(scale-n_scales+1)
            y_values = models_all[sig][scale]
            function(y_values, omegas = omega, ax = axes[sig_rev, scale_rev], lms = lms, log = log, axis_label = False)
            #except:
           #     print('oopsie')
            if sig == 0:
                axes[sig_rev,scale_rev].set_xlabel("Spatial frequency", fontsize= 20)
                #axes[sig_rev,scale].set_xlabel(scale, fontsize=20)
            if scale_rev == 0:
                #axes[sig_rev,scale].set_ylabel(sig,fontsize=20)
                if log:
                    axes[sig_rev,scale_rev].set_ylabel("Power (log10)", fontsize = 20)
                else:
                    axes[sig_rev,scale_rev].set_ylabel("Amplitude", fontsize = 20)
    fig.supxlabel('Input signal (log10)', size = 40)
    fig.supylabel('Output noise →', size = 40)
    fig.suptitle("Temporal frequency = " + str(('{:.3f}').format(fixed_freq)), size = 40)
    