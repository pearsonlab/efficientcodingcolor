#%%

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
import random

from scipy.interpolate import RectBivariateSpline, BivariateSpline




############## IMPORTANT NOTE ###################
#Changed C.interpolate to use Cx_predict instead of Cx. Cx_predict also takes valeus from 0 to 10pi instead of 0 to 2pi







#That looks really nice. Can't have output noise more than 1 after power bug fix 
#test = grid_models([0.125,0.25,0.5,1], [1,2,3,4,5])
#grid_values(test, command = '[0].powers[2]')


#test = grid_models([0.06125,0.125,0.25,0.5,1], [1,2,3,4,5,6]) #To get clean transition through 0


#test = grid_models([10,15,20,25,30,70,100], [10,100,1000,10000])
Cx_name = "Cx_crop512"
predict_Cx_global = True
extrap_global=1
P_total = 1000
eig_labels = ["L+M+S","L+M-S","L-M"]
colors = ['red', 'green', 'blue', 'purple', 'orange', 'brown', 'yellow', 'pink', 'darkred', 'olive']
#Plots on March 17th:
x = 3
#test = grid_models([3,10,30,90,270], [100/(x**2),100/x, 100, 100*x, 100*x*x])

#Biggest one to study output noise:
#test = grid_models([3,10,30,90,270], [100/(x**4),100/(x**3),100/(x**2),100/x, 100, 100*x, 100*x*x])
#test = grid_models([1,3,10,30,90,270], [10,100,1000,2000])

def Cx_error(params, k, Cx_empirical):
    predictions = Cx_theory(params, k)
    return np.sum((np.log10(predictions) - np.log10(Cx_empirical))**2) 
    #return np.sum((predictions - Cx_empirical)**2)

def Cx_theory(params, k):
    A = params[0]
    alpha = params[1]

    return A/(k**alpha)

def fft_wshift(v):
    return np.real(scipy.fft.fftshift(scipy.fft.fft(scipy.fft.ifftshift(v))))

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
    def __init__(self, name, scaling = 5):
        self.C_name = name
        self.C_dict = import_C(self.C_name)
        self.scaling = scaling
        
        self.n_time_freqs = self.C_dict['Cx_eigvals'].shape[1]//2
        self.n_space_freqs = self.C_dict['Cx_eigvals'].shape[0]
        
        self.n_channels = self.C_dict['Cx_bin'].shape[2]
        
        self.Cx_bin = self.C_dict['Cx_bin']/(10**self.scaling)

    def eigen(self, sigin, s_noise = 1):
        
        self.sigin = sigin
        eigvects = np.zeros(self.Cx_bin.shape)
        Cx_pre = np.zeros(self.Cx_bin.shape[:-1])
        
        for i in range(self.Cx_bin.shape[0]):
            for j in range(self.Cx_bin.shape[1]):
                eigs = scipy.linalg.eigh(self.Cx_bin[i,j,:,:], b = np.diag([self.sigin, self.sigin, self.sigin*s_noise])) #np.identity(self.n_channels)*self.sigin)
                Cx_pre[i,j,:] = eigs[0]
                eigvects[i,j,:,:] = eigs[1]
        
        #Cx_pre = self.C_dict['Cx_eigvals'][:,0:self.n_time_freqs,:]
        Cx_pre = Cx_pre[:,0:self.n_time_freqs,:]
        
        Cx_pre = np.flip(Cx_pre, axis = 2)
        
        #U = self.C_dict['Cx_eigvects'][:,0:self.n_time_freqs,:]
        U = eigvects
        
        U = np.flip(U, axis = 3)
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                for c in range(U.shape[3]):
                    if U[i,j,0,c] < 0:
                        U[i,j,:,c] *= -1
        self.Cx = scipy.ndimage.median_filter(Cx_pre, size = 5, axes = 0)
        self.eigvects = np.copy(U)

    def interpolate(self, k, o, i, predict = predict_Cx_global):
        if predict:
            Cx = self.Cx_predict
        else:
            Cx = self.Cx
        func = RectBivariateSpline(np.linspace(0, 2*np.pi, self.Cx.shape[0]), np.linspace(0, 2*np.pi, self.Cx.shape[1]), np.log10(Cx[:,:,i]), s = 0)
        return 10**func(k,o) #November 7th, 2024
    # i is the eigenchannel. Returns eigenvectors number i for different spatial and temporal frequencies. 
    def eigvects_interpolate(self, k, o, i):
        func_L = RectBivariateSpline(np.linspace(0, 2*np.pi, self.eigvects.shape[0]), np.linspace(0, 2*np.pi, self.eigvects.shape[1]), self.eigvects[:,:,0,i], s = 0)
        func_M = RectBivariateSpline(np.linspace(0, 2*np.pi, self.eigvects.shape[0]), np.linspace(0, 2*np.pi, self.eigvects.shape[1]), self.eigvects[:,:,1,i], s = 0)
        func_S = RectBivariateSpline(np.linspace(0, 2*np.pi, self.eigvects.shape[0]), np.linspace(0, 2*np.pi, self.eigvects.shape[1]), self.eigvects[:,:,2,i], s = 0)
        
        #Returns lists of lists but there's no simple way around that if I want the function to generalize for multiple spatial and temporal frequencies.
        return np.array([func_L(k,o), func_M(k,o), func_S(k,o)]) 
    
    def plot_Cx_space(self, n_freqs, channel):
        freqs_rad = np.linspace(0.1, 2*np.pi, n_freqs)
        freqs_index = np.int64(np.linspace(2, self.Cx.shape[1]-1, n_freqs))
        lines = []
        
        for freq_index, freq_rad in zip(freqs_index, freqs_rad):
            Cx = self.Cx[:,freq_index,channel]
            line, = plt.plot(np.log10(Cx), label = "Tf = " + str("{:.2f}".format(freq_rad)))
            lines.append(line)
        plt.legend(handles=lines, fontsize = 30)
        plt.xlabel("Spatial Frequency", size = 30)
        plt.xticks([])
        plt.ylabel("Cx (log10)", size = 30)
        plt.yticks(fontsize=24)
        plt.title(eig_labels[channel], size = 30)
        
    def plot_Cx_ratio(self, n_freqs, channel,skip_first=True):
        if skip_first:
            start = 1
        else:
            start = 0
        freqs_rad = np.linspace(0.1, 2*np.pi, n_freqs)
        freqs_index = np.int64(np.linspace(2, self.Cx.shape[1]-1, n_freqs))
        lines = []
        
        for i in range(start,len(freqs_index)-1):
            Cx1 = self.Cx[:,freqs_index[i],channel]
            Cx2 = self.Cx[:,freqs_index[i+1], channel]
            line, = plt.plot(Cx1/Cx2, label = "Tf: " + str("{:.2f}".format(freqs_rad[i])) + "→" + str("{:.2f}".format(freqs_rad[i+1])))
            lines.append(line)
        plt.legend(handles=lines, fontsize = 30)
        plt.xlabel("Spatial Frequency", size = 30)
        plt.xticks([])
        plt.ylabel("Cx ratio", size = 30)
        plt.yticks(fontsize=24)
        plt.title(eig_labels[channel], size = 30)
    
    def plot_Cx_channel(self,temp_freq):
        lines = []
        plt.figure()
        for channel in range(self.n_channels):
            Cx = self.Cx[:,temp_freq,channel]
            line, = plt.plot(np.log10(Cx), label = "Eigenchannel:" + str(channel))
            lines.append(line)
        plt.legend(handles=lines, fontsize=30)
        plt.xlabel("Spatial Frequency", size = 30)
        plt.ylabel("Cx (log10)", size = 30)
        plt.xticks([])
        plt.axhline(np.max(np.log10(Cx)), color = 'black')
        
    def fit_Cx_theory(self, extrap = 1):
        self.Cx_predict = np.zeros(self.Cx.shape)
        self.A = np.zeros([self.Cx.shape[1], self.Cx.shape[2]])
        self.alpha = np.zeros([self.Cx.shape[1], self.Cx.shape[2]])
                           
        for t in range(C.Cx.shape[1]):
            for c in range(C.Cx.shape[2]):
                k_fit = np.linspace(0.1, np.pi*2, C.Cx.shape[0])
                model = scipy.optimize.minimize(Cx_error, [20000,1.5], method = 'Nelder-Mead', args = (k_fit, C.Cx[:,t,c]))
                k_predict = np.linspace(0.1, np.pi*2*extrap, C.Cx.shape[0])
                
                self.A[t,c] = model.x[0]
                self.alpha[t,c] = model.x[1]
                Cx_predict = Cx_theory(model.x, k_predict)
                self.Cx_predict[:,t,c] = Cx_predict
        
        #plt.plot(np.log10(Cx_theory(you.x,k)))
        #plt.plot(np.log10(C.Cx[:,0,c]))
            
        


C = covariance(Cx_name, 5)

#k = np.linspace(0.1, np.pi*2, C.Cx.shape[0])
#you = scipy.optimize.minimize(Cx_error, [20000,1.5], method = 'Nelder-Mead', args = (k, C.Cx[:,0,c]))
#plt.plot(np.log10(Cx_theory(you.x,k)))
#plt.plot(np.log10(C.Cx[:,0,c]))

class model():
    def __init__(self, fixed_freq, n_freqs, klims, P, fixed_type, sigout, sigin):
        C.eigen(sigin)
        C.fit_Cx_theory(extrap=extrap_global)
        self.sigin = sigin
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
                    ktilde = 1/C.interpolate(self.fixed_freq, self.freqs, i)
                elif self.fixed_type == 'temporal':
                    #ktilde = 1/Cx[:,int(fixed_freq), i]
                    ktilde = 1/C.interpolate(self.freqs, self.fixed_freq, i)
                
                
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
                inter = np.clip(C.interpolate(self.fixed_freq,indices, channel), 1e-32, np.inf).T
                ktilde = 1/inter
            elif self.fixed_type == 'temporal':
                inter = np.clip(C.interpolate(indices, self.fixed_freq, channel), 1e-32, np.inf)
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
            CC = np.minimum(C.interpolate(k, o, i), 1e32)
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
            ktilde = 1/C.interpolate(self.fixed_freq, indices, channel)
        elif self.fixed_type == 'temporal':
            ktilde = 1/C.interpolate(indices, self.fixed_freq, channel)
        
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
        self.freqs_interp = np.linspace(0,np.pi*2,100000)
        #self.freqs_interp = self.freqs
        vspace_omega = []; vv_all = []; vf_all = []; vvf_all = []; log_va_all = []
        if self.log_nus is not None:
            for i in range(self.n_channels):
                self.filter_power(10**self.log_nus[i], self.freqs, None, i, sum_power = False)
                vv = self.filter_k(10**self.log_nus[i], i)(self.freqs_interp)
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
            eigvects_interpolate = C.eigvects_interpolate(self.freqs_interp,self.fixed_index,eig_c)[:,:,0]
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
            
            vv_lms += (vv_interp*eigvects_interpolate)
            vf_lms += (vf_interp*eigvects_interpolate)


        vspace_LMS = []
        self.vf_lms = vf_lms
        self.vv_lms = vv_lms
        for color in range(n_channels):
            vvf = pad_and_reflect(vf_lms[color,:], vf_lms.shape[1]*2)
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
    fixed_freqs = np.linspace(0.1,2*np.pi, n_fixed_freq)
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

def plot_filters(models, ax = None, plot_range = 50, lms = False, omegas = 'all', n_channels = 3):
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
    plt.xlabel("Input noise", size = 30)
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
            line, = ax.plot(model.freqs_interp, np.log10(vf[c,:]*np.conjugate(vf[c,:])), color = colors[c], label = labels[c])
        else:
            line, = ax.plot(model.freqs_interp, vf[c,:], color = colors[c], label = labels[c])
            ax.axhline(0)
        #ax.axvline(2.5)
        lines.append(line)
        #line, = ax[i].plot(x_values, y_values, color = colors[omega], label = str(('{:.3f}').format(model.fixed_freq)))
    if axis_label:
        ax.legend(handles=lines)
        ax.set_xlabel("Frequency", size = 30)
        ax.set_ylabel("Power", size = 30)
        ax.set_title("sigout = " + str(models[0].sigout) + ", sigin = " + str(models[0].sigin))


def plot_filters_simple(models, omegas, plot_range = 50, ax = None, lms = False, y_range = None, log = False, axis_label = False):
    
    model = models[omegas]
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
    for i in range(model.n_channels):
        if model.fixed_type == 'temporal':
            y_values.extend(vspace[i,center-plot_range:center+plot_range])

    ax.plot(y_values, color = colors[omegas], label = str(('{:.3f}').format(model.fixed_freq)))
    ax.axvline(plot_range*2)
    ax.axvline(plot_range*4)
    if y_range is not None:
        ax.set_ylim([y_range[0], y_range[1]])
    


#My input is a function and a set of models at diff params
def grid_plots(function, models_all, omega, lms, log = False, sharey = True, size_mul = 0.5):
    n_sigs = len(models_all)
    n_scales = len(models_all[0])
    fixed_freq = models_all[1][1][omega].fixed_freq
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
                
                function(y_values, omegas = omega, ax = axes[sig_rev, scale], lms = lms, log = log, axis_label = False)
                #except:
               #     print('oopsie')
            else:
                sig_value = 0
                scale_value = 0
            if sig == 0 and scale_rev == 0:
                print('hi')
                axes[sig_rev,scale_rev].set_xlabel(x_label, fontsize= 20*size_mul)
                #axes[sig_rev,scale].set_xlabel(scale, fontsize=20)
                if log:
                    axes[sig_rev,scale_rev].set_ylabel("Power (log10)", fontsize = 20*size_mul)
                else:
                    axes[sig_rev,scale_rev].set_ylabel("Amplitude", fontsize = 20*size_mul)
            axes[sig_rev,scale].set_title("sigout = " + str(('{:.1f}').format(sig_value)) + ", sigin = " + str(('{:.1f}').format(scale_value)), size = 30*size_mul)
            axes[sig_rev,scale].set_xticks([])
            axes[sig_rev,scale].set_yticks([])
    fig.supxlabel('Input noise →', size = 40*size_mul)
    fig.supylabel('Output noise →', size = 40*size_mul)
    fig.suptitle("Temporal frequency = " + str(('{:.3f}').format(fixed_freq)) + ", power = " + str(P_total), size = 40*size_mul)
    