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

from scipy.interpolate import RectBivariateSpline, BivariateSpline


def soft_bandpass(lo, hi, freqs, stiffness=10):
    if lo <= 0:
        return scipy.special.expit(stiffness * (hi - freqs))
    else:
        return scipy.special.expit(stiffness * (freqs - lo)) * scipy.special.expit(stiffness * (hi - freqs))


def import_C():
    with open(os.getcwd() + "/../../Cx.pkl", 'rb') as f:
        x = pickle.load(f)
    return x

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
        
    def interpolate(self, k, o, i, dist = 0.01):
        func = RectBivariateSpline(np.linspace(0, 2*np.pi, self.Cx.shape[0]), np.linspace(0, 2*np.pi, self.Cx.shape[1]), np.log10(self.Cx[:,:,i]), s = 0)
        return 10**func(k,o)/1000000000 #November 7th, 202
#return 10**func(k,o)/1000000

C = covariance()

class model():
    def __init__(self, fixed_freq, n_freqs, klims, P, fixed_type, sigout):

        self.sigout = sigout
        self.fixed_freq = fixed_freq
        self.freqs = np.linspace(0,2*np.pi, n_freqs)
        self.P = P
        self.fixed_type = fixed_type
        power_constraint = {'type': 'ineq', 'fun': self.excess_power, 'args': (fixed_freq, self.freqs, klims, P, fixed_type)}
        eps_fun = self.create_information_eps()
        res = opt.minimize(eps_fun, np.array([-10,-10,-10]), bounds=[(-16, np.inf)], constraints=[power_constraint])
        if res.success:
            #print("Excess power: ", excess_power(res.x, fixed_freq, indices, klims, P, fixed_type))
            return res.x
        else:
            print("Optimizer failed to converge!")
            return None
        self.log_nus = res.x
        
    def create_information_eps(self, return_sum = True):
        def information_eps(log_eps):
            infos = []
            for i in range(C.n_channels):
                eps_eigenchannel = 10**log_eps[i]
                if self.fixed_type == 'spatial':
                    #ktilde = 1/Cx[int(fixed_freq),:, i]
                    ktilde = 1/C.interpolate(int(self.fixed_freq), self.freqs, i)
                elif self.fixed_type == 'temporal':
                    #ktilde = 1/Cx[:,int(fixed_freq), i]
                    ktilde = 1/C.interpolate(self.freqs, int(self.fixed_freq), i)
                ktilde = self.pad_and_reflect(ktilde[:,0],ktilde.shape[0]*2 - 2)
                #numer = np.maximum(0, sigout**2 /(2 * (ktilde + 1e-32)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * sigout**2)) - 1) - sigout**2) + sigout**2
                #denom = np.maximum(0, sigout**2 /(2 * (ktilde + 1)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * sigout**2)) + 1) - sigout**2) + sigout**2
    
                numer = np.maximum(0, 1/(2 * (ktilde + 1e-32)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * self.sigout**2)) - 1) - 1) + 1
                denom = np.maximum(0, 1/(2 * (ktilde + 1)) * (np.sqrt(1 + 4 * ktilde/(eps_eigenchannel * self.sigout**2)) + 1) - 1) + 1
                
                
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
                inter = np.clip(C.interpolate(int(self.fixed_freq),indices, channel), 1e-32, np.inf).T
                ktilde = 1/inter
            elif self.fixed_type == 'temporal':
                inter = np.clip(C.interpolate(indices, int(self.fixed_freq), channel), 1e-32, np.inf)
                ktilde = 1/inter
            ktilde = self.pad_and_reflect(ktilde[:,0],ktilde.shape[0]*2 - 2)
            sqrt_piece = np.sqrt(1 + (4/eps) * ktilde)
            
            v2 = 0.5 * (sqrt_piece + 1) / (1 + ktilde) - 1
            v2 = np.sqrt(np.maximum(v2, 0) * self.sigout**2)
            if k_lims:
                unit_cell_k = 0# soft_bandpass(k_lims[0], k_lims[1], k)
                v2 *= unit_cell_k
            return v2
        return v_opt
    
    def filter(self, nu, k_lims=None):
        def v_opt(k, o, i):
            CC = np.minimum(C.interpolate(k, o, i), 1e32)
            sqrt_piece = np.sqrt(CC**2 + (4/nu) * (1/self.sigout**2) * CC)
            v2 = 0.5 * (sqrt_piece + CC) / (1 + CC) - 1
            v2 = np.sqrt(np.maximum(v2, 0) * self.sigout**2)
            if k_lims:
                unit_cell_k = 0#soft_bandpass(k_lims[0], k_lims[1], k)
                v2 *= unit_cell_k
    
            return v2 #- np.min(v2)
        return v_opt
    
    def pad_and_reflect(filter, N, padval=0):
        """
        Zero-pad to length N//2 and reflect about origin to make length N. 
        """
        expanded = np.pad(filter, (0, N//2 + 1 - len(filter)), constant_values=(0, padval))
        return np.concatenate([expanded[::-1], expanded[1:]])
    
    def extrap_and_reflect(filter, N, return_log=False):
        """
        Linearly extrapolate the log filter to better approximate tails and reflect 
        about origin to make length N.
        """
        extrap_fun = interp1d(range(0, len(filter)), np.maximum(-32, np.log(filter)), fill_value='extrapolate')
        expanded = extrap_fun(range(0, N//2 + 1))
        if not return_log:
            expanded = np.exp(expanded)
        return np.concatenate([expanded[::-1], expanded[1:]])
    
    def filter_power(self, eps, indices, klims, channel):
        vfun = self.filter_k(eps, channel, klims)
        v2 = vfun(indices)
        if self.fixed_type == 'spatial':
            ktilde = 1/C.interpolate(int(self.fixed_freq), indices, channel)
        elif self.fixed_type == 'temporal':
            ktilde = 1/C.interpolate(indices, int(self.fixed_freq), channel)
        
        ktilde = self.pad_and_reflect(ktilde[:,0],ktilde.shape[0]*2 - 2)
    
        dk = 2*np.pi/len(indices)
    
    
        freqs = self.pad_and_reflect(self.freqs, len(self.freqs)*2 - 2)
    
        return np.sum(v2**2 * np.abs(freqs) * (1/ktilde + 1) * dk)/(2 * np.pi)
        #John: Needs to make sure I do the conversion from integral to sum correctly
        #Need to make srue dk is the correct value. dk = 2*pi/L
        #By convention, frequencies discrete are -pi to pi. My frequencies are from 0 to L. 

    def excess_power(self, log_eps, indices, klims):
        total_power = 0
        for i in range(C.n_channels):
            eig_power = self.filter_power(10**log_eps[i], indices, klims, i)
            total_power += eig_power
        return self.P - total_power
    
    def get_vspace(self):
        # make a double exponential smoothing filter
        fz = np.arange(-.2,.2, 0.01)
        ff = np.exp(-20 * np.abs(fz))
        ff /= np.sum(ff)
        
        vspace_omega = []; vv_all = []; vf_all = []; vvf_all = []
        for i in range(self.n_channels):
            vv = self.filter_k(10**self.log_nus[i], i)(self.freqs)
            vv = vv[vv.shape[0]//2:]
            vf = np.convolve(vv, ff, mode = 'same')
            vvf = self.pad_and_reflect(vf, vf.shape[0]*2)
            

            vspace = np.real(scipy.fft.fft(vvf))
            vspace = np.real(scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(vvf))))
            vspace_omega.append(vspace); vv_all.append(vv); vf_all.append(vf); vvf_all.append(vvf)
        vspace_omega = np.array(vspace_omega)
        vspace_omega /= np.linalg.norm(vspace_omega)
        self.vspace = vspace_omega; self.vv = vv_all; self.vf = vf_all; self.vvf = vvf_all
        
def spatial_filters(n_temp_freqs, n_space_freqs, sigout):
    temp_freqs = np.linspace(0,np.pi*2, n_temp_freqs)
    models = []
    for temp_freq in temp_freqs:
        mod = model(temp_freq, n_space_freqs, None, 1, 'temporal', sigout)
        mod.get_vspace()
        models.append(mod)
    return models

def plot_spatial_filters(models):
    fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'yellow', 'pink', 'darkred', 'olive']
    channel_labels = ['L+M+S', 'L+M-S', 'L-M']
    lines = []
    for omega in len(models):
        model = models[omega]
        vspace = model.vspace
        for i in range(C.n_channels):
            if np.max(models[omega].vspace[i,:]) == 0:
                print("Zero vspace for kf: ", omega, i)
            #line, = ax[i].plot(zz, vspace, color = colors[c], label = str(('{:.3f}').format(fixed_omega)))
            line, = ax[i].plot(vspace[i,:], color = colors[omega], label = str(('{:.3f}').format(model.fixed_omega)))
            
            center = int(vspace.shape[0]/2)
            plot_range = 30
            ax[i].set_xlim(center - plot_range, center + plot_range)   
            #ax[i].set_yticks([])
            ax[i].set_xlabel(r"$z$")
            ax[i].set_title(r"Spatial filter $v(z)$ for " + channel_labels[i]);
            #else:
               # print("Filter was zero for channel " + str(i) + " at temporal frequency " + str(fixed_omega))
        lines.append(line)   
    ax[C.n_channels-1].legend(handles=lines, title = "Temporal frequency", fontsize = 12)
            