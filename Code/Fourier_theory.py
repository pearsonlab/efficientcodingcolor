#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 13:10:21 2025

@author: david
"""

import os
import numpy as np
import pickle
import scipy
import scipy.optimize as opt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.signal as ssig
import random
import math

from scipy.interpolate import RectBivariateSpline, BivariateSpline

############## IMPORTANT NOTE ###################
#Changed C.interpolate to use Cx_predict instead of Cx. Cx_predict also takes valeus from 0 to 10pi instead of 0 to 2pi

#That looks really nice. Can't have output noise more than 1 after power bug fix 
#test = grid_models([0.125,0.25,0.5,1], [1,2,3,4,5])
#grid_values(test, command = '[0].powers[2]')
#test = grid_models([0.06125,0.125,0.25,0.5,1], [1,2,3,4,5,6]) #To get clean transition through 0
#test = grid_models([10,15,20,25,30,70,100], [10,100,1000,10000])
#Good range:
#test = grid_models([0.3,1,3,10,30,100], [30,100,300,1000,10000,100000])

correlated_k_noise = False
extrap_global=100
n_samples_mul = 10


apply_mtf = True
k0_global = 2*np.pi*extrap_global / 10

LM_corr = 0
s_noise_global = 1

min_k_freq = 0
P_total = 1000
lms_labels = ['Long', 'Medium', 'Short']
eig_labels = ["L+M+S","L+M-S","L-M"]
eig_colors = ['orangered', 'darkorange', 'gold']
colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'yellow', 'pink', 'darkred', 'olive']
norm_freq_range=[min_k_freq,2*np.pi*extrap_global]

#Plots on March 17th:
x = 3
#test = grid_models([3,10,30,90,270], [100/(x**2),100/x, 100, 100*x, 100*x*x])

#Biggest one to study output noise:
#test = grid_models([3,10,30,90,270], [100/(x**4),100/(x**3),100/(x**2),100/x, 100, 100*x, 100*x*x])
#test = grid_models([1,3,10,30,90,270], [10,100,1000,2000])

def fft_wshift(v):
    return np.real(scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(v))))

def soft_bandpass(lo, hi, freqs, stiffness=10):
    if lo <= 0:
        return scipy.special.expit(stiffness * (hi - freqs))
    else:
        return scipy.special.expit(stiffness * (freqs - lo)) * scipy.special.expit(stiffness * (hi - freqs))

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

#Taylor approximation for when k << 1. 
def Taylor_small1(freqs, k0, A, sigout, nu, sigin, delta = 1.4, alpha = 1.3):
    freqs = freqs + 0.001
    if apply_mtf:
        mtf = math.e**(-(freqs/k0)**delta)
    else:
        mtf = 1
    lamb = A/(freqs**alpha * sigin)  * mtf
    inner = lamb/8 * np.sqrt(lamb * nu * sigout**2) - 1
    return sigout**2 * freqs * np.clip(inner, a_min = 0, a_max = np.inf)

#Compares approximation with k << 1 with the original. Doesnt work
def power_predict2_with_t(freqs, k0, A, sigout, nu, delta = 1.4, alpha = 1.3):
    freqs = freqs + 0.001
    t = 0.5*(freqs/k0)**delta
    e = math.e
    kf = A**(1/alpha)
    p_pre = t**(2-(alpha/2) - delta)*e**(-t)
    p = nu**0.5 * sigout**2 * np.clip(kf**(alpha/2) * 2**((2-alpha/2)/delta) * k0**(2-alpha/2) * p_pre - freqs, a_min = 0, a_max = np.inf)
    return p

#Same as power_predict1 but before substitution
def power_predict2(freqs, k0, A, sigout, nu, sigin, delta = 1.4, alpha = 1.3):
    freqs = freqs + 0.001
    if apply_mtf:
        mtf = math.e**(-(freqs/k0)**delta)
    else:
        mtf = 1
    lamb = A/(freqs**alpha *sigin)  * mtf
    dk = 2*np.pi/len(freqs)
    return sigout**2 * freqs * np.clip(np.sqrt(lamb/(nu*sigout**2)) - 1, 0, np.inf) * dk / (2*np.pi)

def power_fun(freqs, k0, A, sigout, nu, sigin, delta = 1.4, alpha = 1.3):
    freqs = freqs + 0.001
    mtf = math.e**(-(freqs/k0)**delta)
    lamb = A/(freqs**alpha * sigin)  * mtf
    
    sqrt_block = np.sqrt(1 + 4/(lamb*nu*sigout**2))
    square_block = np.clip(lamb/2 * (sqrt_block - 1) - 1, 0, np.inf)
    dk = 2*np.pi/len(freqs)
    r = sigout**2 * freqs * square_block * dk / (2*np.pi)
    return r

    #C.power_predict1(k0_global)
#k = np.linspace(0.1, np.pi*2, C.Cx.shape[0])
#you = scipy.optimize.minimize(Cx_error, [20000,1.5], method = 'Nelder-Mead', args = (k, C.Cx[:,0,c]))
#plt.plot(np.log10(Cx_theory(you.x,k)))
#plt.plot(np.log10(C.Cx[:,0,c]))

class model():
    def __init__(self, omega, n_freqs, klims, P, fixed_type, sigout, sigin, step_scaling = 1):
        #C.create_Cx([100,50,25])
    
        self.freqs = np.linspace(norm_freq_range[0],norm_freq_range[1], n_freqs*n_samples_mul)
        self.A = [100,10,1]
        self.n_channels = len(self.A)
        self.omega = omega
        nu_init = -10 * step_scaling
        self.Cx_maker(self.A)
        if apply_mtf:
            self.MTF(k0_global)
            #C.MTF(1, predict=True)
        self.Cx = self.Cx / sigin
        
        
        self.sigin = sigin
        self.sigout = sigout
        self.fixed_freq = omega
        
        
        self.P = P
        self.fixed_type = fixed_type
        self.n_channels = self.n_channels
        self.power_freqs = []
        self.n_freqs = n_freqs
        self.step_scaling = step_scaling
        
        if fixed_type == 'temporal':
            self.fixed_index = int(omega*n_freqs/(2*np.pi))
            self.n_space_freqs = n_freqs
        power_constraint = {'type': 'ineq', 'fun': self.excess_power, 'args': (self.freqs, klims)}
        eps_fun = self.create_information_eps()
        res = opt.minimize(eps_fun, np.array([nu_init]), bounds=[(-20, np.inf)], constraints=[power_constraint])
        if res.success and res.x != nu_init:
            #print("Excess power: ", excess_power(res.x, fixed_freq, indices, klims, P, fixed_type))
            self.log_nu = res.x/self.step_scaling
        else:
            if not res.success:
                print("Optimizer failed to converge!")
            if res.x == nu_init:
                print("Error: Nu is still the same as initialized.")
            #self.log_nu = None
            self.log_nu = res.x
    def Cx_maker(self, A, alpha = 1.3):
        self.alpha = alpha
        eigvects = np.array([[np.sqrt(1/3),np.sqrt(1/3),np.sqrt(1/3)],
                                  [-np.sqrt(1/4), -np.sqrt(1/4), np.sqrt(1/2)],
                                  [np.sqrt(1/2), -np.sqrt(1/2), 0]])
        #eigvects = np.repeat(eigvects[np.newaxis,:,:], self.n_space_freqs*n_samples_mul, axis=0)
        self.eigvects= eigvects #np.repeat(eigvects[:,np.newaxis,:,:], self.n_time_freqs, axis = 1)
        
        
        
        k_freqs = self.freqs
        Cx = np.zeros([len(self.freqs), self.n_channels])
        for k in range(len(self.freqs)):
            for c in range(self.n_channels):
                Cx[k,c] = A[c]/((k_freqs[k]+0.001)**alpha * self.omega**2)
        
        self.Cx = Cx
        self.Cx_predict = Cx
        self.A = A

    def MTF(self, k0, delta = 1.4):
        k = self.freqs
        mtf = np.exp(-(k/k0)**delta)
        mtf = np.expand_dims(mtf, 1)
        mtf = np.repeat(mtf, 3, axis = 1)
        #mtf = np.repeat(mtf, self.Cx.shape[1], axis = 1)
        self.mtf = mtf
        self.Cx = self.Cx * self.mtf    
    def create_information_eps(self, return_sum = True):
        def information_eps(log_nu):
            infos = []
            info_freqs = []
            for i in range(self.n_channels):
                eps_eigenchannel = 10**(log_nu/self.step_scaling)
                vfun = self.filter_k(eps_eigenchannel, i)
                #ktilde = 1/Cx[:,int(fixed_freq), i]
                ktilde = 1/self.Cx[:, i]
                
                
                power = self.filter_power(eps_eigenchannel, None, i)
                ktilde = pad_and_reflect(ktilde,ktilde.shape[0]*2 - 2)
                len_tilde = ktilde.shape[0]
                
                numer_new = vfun(self.freqs)**2 * (1/ktilde + 1) + self.sigout**2
                denum_new = vfun(self.freqs)**2 + self.sigout**2
                numer = power + self.sigout**2 #Old version
                denom = (ktilde/(1+ktilde)*power + self.sigout**2) #I think the bug is here :) 
                
                self.info_numer = numer
                self.info_denom = denom
                self.numer_new = numer_new
                self.denum_new = denum_new
                
                self.ktilde = ktilde
                
                self.power_test = power
                info_freqs.append(np.log(numer_new) - np.log(denum_new))
                
                
                info = np.sum(np.log(numer_new) - np.log(denum_new))
                infos.append(info)
            self.info_freqs = info_freqs
            if return_sum:
                return np.sum(infos)*-1
            else:
                return infos 
        return information_eps

    def filter_k(self, eps, channel, k_lims=None):
        def v_opt(indices):
            inter = self.Cx[:, channel].T
            lamb = pad_and_reflect(inter,inter.shape[0]*2 - 2)
            #ktilde = 1/inter
            #ktilde = pad_and_reflect(ktilde,ktilde.shape[0]*2 - 2)
            sqrt_piece = np.sqrt(1 + (4/(eps*self.sigout**2 * lamb)))
            self.sqrt_piece = sqrt_piece
            #print(sqrt_piece.shape, lamb.shape)
            v2 = 0.5 * (sqrt_piece + 1) / (1 + 1/lamb) - 1
            v2 = np.sqrt(np.maximum(v2, 0) * self.sigout**2)
            if k_lims:
                unit_cell_k = 0# soft_bandpass(k_lims[0], k_lims[1], k)
                v2 *= unit_cell_k
            return v2
        return v_opt
    #Approximation for when lambda << 1.
    def filter_k_approx(self, eps, channel):
        def v_opt(indices):
            #print(indices)
            lambda_pre = self.Cx[:, channel].T
            lam = pad_and_reflect(lambda_pre,lambda_pre.shape[0]*2 - 2)
            v2 = np.sqrt(lam/(eps*self.sigout**2)) - 1
            v2 = np.maximum(v2,0)*self.sigout**2
            return v2
        return v_opt
    
    def filter(self, nu, k_lims=None):
        def v_opt(k, channel):
            CC = np.minimum(self.Cx[k, channel], 1e32)
            sqrt_piece = np.sqrt(CC**2 + (4/nu) * (1/self.sigout**2) * CC)
            v2 = 0.5 * (sqrt_piece + CC) / (1 + CC) - 1
            v2 = np.sqrt(np.maximum(v2, 0) * self.sigout**2)
            if k_lims:
                unit_cell_k = 0#soft_bandpass(k_lims[0], k_lims[1], k)
                v2 *= unit_cell_k
    
            return v2 #- np.min(v2)
        return v_opt
 
    def filter_power(self, eps, klims, channel, append = True):
        indices = self.freqs
        vfun = self.filter_k(eps, channel, klims)
        v2 = vfun(indices)
        v2_approx = self.filter_k_approx(eps,channel)(indices)
        #eigen = self.Cx[:, self.fixed_freq, channel]
        eigen = self.Cx[:,channel]
        eigen = pad_and_reflect(eigen,eigen.shape[0]*2 - 2)
    
        dk = 2*np.pi/len(indices)
    
    
        freqs = pad_and_reflect(self.freqs, len(self.freqs)*2 - 2)
        self.v2 = v2
        self.v2_approx = v2_approx**2 * np.abs(freqs) * dk
        #print(v2.shape, freqs.shape, ktilde.shape)
        power_freqs = v2**2 * np.abs(freqs) * (eigen + 1) * dk
        #Double check that freqs and ktilde are evaluated at the same k. 
        if append: 
            #return np.sum(power_freqs)/(2*np.pi)
            return power_freqs/(2*np.pi)
        else:
            power_freqs = power_freqs/(2*np.pi)
            self.power_freqs.append(power_freqs)
            #return power_freqs
        #return np.sum(v2**2 * np.abs(freqs) * (1/ktilde + 1) * dk)/(2 * np.pi)
        #John: Needs to make sure I do the conversion from integral to sum correctly
        #Need to make srue dk is the correct value. dk = 2*pi/L
        #By convention, frequencies discrete are -pi to pi. My frequencies are from 0 to L. 

    def excess_power(self, log_nu, indices, klims):
        total_power = 0
        eig_powers = []
        for i in range(self.n_channels):
            eig_power = self.filter_power(10**(log_nu/self.step_scaling), klims, i)
            total_power += eig_power
            eig_powers.append(eig_power)
        self.powers = eig_powers
        return self.P - total_power
     
    def get_v(self):
        n_freqs_interp = 100000
        # make a double exponential smoothing filter
        fz = np.linspace(-.2,.2, int(n_freqs_interp/100)) #0.1 of freqs are smoothing 
        #fz = np.arange(-.2,.2, 0.0001)
        ff = np.exp(-20 * np.abs(fz))
        ff /= np.sum(ff)
        self.freqs_interp = self.freqs#np.linspace(0,np.pi*2*extrap_global,n_freqs_interp)
        #self.freqs_interp = self.freqs
        vspace_omega = []; vv_all = []; vf_all = []; vvf_all = []; log_va_all = []
        if self.log_nu is not None:
            for i in range(self.n_channels):
                self.filter_power(10**self.log_nu, None, i, append = False)
                vv = self.filter_k(10**self.log_nu, i)(self.freqs_interp)
                vv = vv[vv.shape[0]//2:]
                vf = vv
                #vf = np.convolve(vv, ff, mode = 'same')
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
            eigvects_interpolate = self.eigvects
            vv_expand = np.repeat(vv[eig_c,:,np.newaxis],3,1).T
            vf_expand = np.repeat(vf[eig_c,:,np.newaxis],3,1).T
            #print(vv_expand.shape, len(self.freqs_interp))
            vv_interp = np.zeros([vv_expand.shape[0], len(self.freqs_interp)])
            vf_interp = np.zeros([vf_expand.shape[0], len(self.freqs_interp)])
            for c in range(vv_expand.shape[0]):
                
                vv_interp[c,:] = np.interp(self.freqs_interp, np.linspace(0,np.pi*2,vv_expand.shape[1]), vv_expand[c,:])
                vf_interp[c,:] = np.interp(self.freqs_interp, np.linspace(0,np.pi*2,vv_expand.shape[1]), vf_expand[c,:])
                
            #vv_lms += (vv_expand*eigvects_interpolate)
            #vf_lms += (vf_expand*eigvects_interpolate)
            #print(eigvects_interpolate.shape)
            flips1 = random.choices([1,-1], k = len(self.freqs_interp))
            flips2 = random.choices([1,-1], k = len(self.freqs_interp))
            
            #if eig_c == 1:
            #    eigvects_interpolate *= flips1
            #if eig_c == 2:
                #this1 = 1
            #    eigvects_interpolate *= flips2
            
            vv_lms += (eigvects_interpolate@vv_interp)
            vf_lms += (eigvects_interpolate@vf_interp)


        vspace_LMS = []
        self.vf_lms = vf_lms
        self.vv_lms = vv_lms
        for color in range(n_channels):
            vvf = pad_and_reflect(vf_lms[color,:], vf_lms.shape[1]*2)#  * 4) #Add this piece if you want smooth filters without correlated LM noise, just MTF. 
            #print('hi')
            self.vvf_test = vvf
            #vvf_interp = np.interp(np.linspace(0,np.pi*2,1000000), np.linspace(0,np.pi*2,vvf.shape[0]), vvf)
            #print('hi again')
            vspace = np.real(scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(vvf))))
            #self.vvf_interp = vvf_interp
            vspace_LMS.append(vspace)
        vspace_LMS /= np.linalg.norm(np.array(vspace_LMS))
        self.vspace_LMS = vspace_LMS
       
def train_filters(n_fixed_freq, n_freqs, fixed_type, sigout, sigin):
    fixed_freqs = np.array([1])
    #fixed_freq = np.linspace(0, np.pi/4, 8)
    models = []
    for fixed_freq in fixed_freqs:
        #print(fixed_freq)
        mod = model(fixed_freq, n_freqs, None, P_total, fixed_type, sigout, sigin)
        
        success = mod.get_v()
        if success:
            mod.lms_RFs()
            models.append(mod)
        else:
            models.append(None)
    return models

def plot_filters(models, ax = None, plot_range = 20, lms = False, omegas = 'all', n_channels = 3):
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(16, 8))
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'yellow', 'pink', 'darkred', 'olive']
    if not lms:
        channel_labels = ['L+M+S', 'L+M-S', 'L-M']
    else:
        channel_labels = ['L', 'M', 'S']
    lines = []
    x_length = models[0].vspace_LMS.shape[1]
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
        
        for i in range(n_channels):
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
    ax[n_channels-1].legend(handles=lines, title = label_title, fontsize = 25, title_fontsize=20)
    for axis in ax:
        axis.set_ylim([ymin, ymax])
#THERE IS A BUG WITH ARGMAX PLEASE FIX IT BEFORE USE!!!! THERE IS A BUG WITH ARGMAX PLEASE FIX IT BEFORE USE!!!! THERE IS A BUG WITH ARGMAX PLEASE FIX IT BEFORE USE!!!! THERE IS A BUG WITH ARGMAX PLEASE FIX IT BEFORE USE!!!!

            
def grid_models(sigs, scales, n_freqs = 10000, n_omegas = 4):
    models_all = []
    for i in range(len(sigs)):
        models_sig = []
        for j in range(len(scales)):
            models = train_filters(n_omegas, n_freqs, 'temporal', sigs[i], scales[j])
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
    plt.xlabel("Input noise", size = 30)
    plt.xticks([])
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)

    plt.title("Power in L-M eigenchannel at the lowest temporal frequency", size = 30)
    return grid


#Frequency plot for specific temporal frequency, input and output noise levels 
def plot_freqs(models, omegas, log = False, ax = None, lms = False, axis_label = True, ratio = 1):
    model = models[omegas]
    if ax is None:
        fig, ax = plt.subplots(1,1)
    lines = []
    
    n_points = len(model.freqs_interp)
    if lms:
        labels = ['L', 'M', 'S']
        colors = ['red', 'green', 'blue']
        vf = model.vf_lms
    else:
        labels = ['L+M+S', 'L+M-S', 'L-M']
        colors = ['orangered', 'darkorange', 'gold']
        vf = model.vf
    for c in range(model.n_channels):
        if log:
            freqs = model.freqs_interp[0:int(n_points*ratio)]
            line, = ax.plot(freqs, np.log10(vf[c,0:int(n_points*ratio)]), color = colors[c], label = labels[c])
            
            ax.plot(freqs, np.log10(power_fun(freqs, k0_global, A = model.A[c], sigout = model.sigout, sigin = model.sigin, nu = 10**model.log_nu)), color = colors[c], linestyle = 'dashed')
        else:
            freqs = model.freqs_interp[0:int(n_points*ratio)]
            line, = ax.plot(freqs, vf[c,0:int(n_points*ratio)], color = colors[c], label = labels[c])
            
            ax.plot(freqs, power_predict2(freqs, k0_global, A = model.A[c], sigout = model.sigout, nu = 10**model.log_nu, sigin=model.sigin), color = colors[c], linestyle = 'dashed')
            #ax.axhline(0)
        #ax.axvline(2.5)
        lines.append(line)
        
        #line, = ax[i].plot(x_values, y_values, color = colors[omega], label = str(('{:.3f}').format(model.fixed_freq)))
    if axis_label:
        ax.legend(handles=lines,fontsize=12)
        ax.set_xlabel("Spatial frequency", size = 20)
        ax.set_ylabel("Power", size = 20)
        #ax.set_title("sigout = " + str(models[0].sigout) + ", sigin = " + str(models[0].sigin))


def plot_filters_simple(models, omegas, plot_range = 100, ax = None, lms = False, y_range = None, log = False, axis_label = False, points = False):
    model = models[omegas]
    lms_colors = ['r', 'g', 'b']
    if ax is None:
        fig, ax = plt.subplots(1,1)
    x_length = model.vspace_LMS.shape[1]
    center = x_length//2
        
    if lms:
        vspace = model.vspace_LMS
    else:
        vspace = model.vspace
    #y_values = np.zeros([vspace.shape])
    y_values = []
    y_colors = []
    start = 0
    if points:
        marker = 'o'
    else:
        marker = ','
    for i in range(model.n_channels):
        if model.fixed_type == 'temporal':
            ax.plot(np.arange(start, start + plot_range*2), vspace[i,center-plot_range:center+plot_range], color = lms_colors[i], label = str(('{:.5f}').format(model.fixed_freq)), marker = marker, markersize = 3)
            start += plot_range*2
            
    ax.axis('off')
    #ax.axvline(plot_range*2)
    #ax.axvline(plot_range*4)
    ax.axhline(0, linestyle = '--')
    
    if y_range is not None:
        ax.set_ylim([y_range[0], y_range[1]])
    


#My input is a function and a set of models at diff params
def grid_plots(function, models_all, omega, lms, log = False, sharey = True, size_mul = 0.5, title = ""):
    n_sigs = len(models_all)
    n_scales = len(models_all[0])
    flag = False
    for i in range(len(models_all)):
        for j in range(len(models_all[0])):
            model_ex = models_all[i][j][omega]
            if model_ex is not None:
                break
                flag = True
        if flag:
            break

    fixed_freq = model_ex.fixed_freq
    fig, axes = plt.subplots(n_sigs, n_scales, sharey = sharey)
    
    if function == plot_freqs:
        x_label = 'Spatial Frequency'
    elif function == plot_filters_simple:
        x_label = 'Space'
    else:
        x_label = ''
    
    for sig in range(n_sigs):
        for scale in range(n_scales):
            model = models_all[sig][scale]
            sig_rev = np.abs(sig-n_sigs+1)
            scale_rev = np.abs(scale-n_scales+1)
            if model[omega] is not None:
                
                
                y_values = models_all[sig][scale]
                sig_value = model[omega].sigout
                scale_value = model[omega].sigin
                
                function(y_values, omegas = omega, ax = axes[sig_rev, scale], lms = lms, log = log, axis_label = True)
                #except:
               #     print('oopsie')
            else:
                sig_value = 0
                scale_value = 0
            if sig == 0 and scale_rev == 0 and False:
                print('hi')
                axes[sig_rev,scale_rev].set_xlabel(x_label, fontsize= 20*size_mul)
                #axes[sig_rev,scale].set_xlabel(scale, fontsize=20)
                if log:
                    axes[sig_rev,scale_rev].set_ylabel("Power (log10)", fontsize = 20*size_mul)
                else:
                    axes[sig_rev,scale_rev].set_ylabel("Amplitude", fontsize = 20*size_mul)
            #axes[sig_rev,scale].set_title("sigout = " + str(('{:.1f}').format(sig_value)) + ", sigin = " + str(('{:.3f}').format(scale_value)), size = 30*size_mul)
            if model[omega] is not None and False:
                axes[sig_rev,scale].set_title(str("{:.3f}".format(model[omega].log_nu)) + ", " + str("{:.3f}".format(model[omega].log_nu)) + ", " + str("{:.3f}".format(model[omega].log_nu)))
            axes[sig_rev,scale].set_xticks([])
            axes[sig_rev,scale].set_yticks([])
    fig.supxlabel(r'$\leftarrow$ $\frac{C_x}{\sigma_{\mathrm{in}}^2}$', size = 80*size_mul)
    fig.supylabel(r'$\leftarrow$ $\frac{P}{\sigma_{\mathrm{out}}^2}$', size = 80*size_mul)
    fig.suptitle(title, size = 80*size_mul)