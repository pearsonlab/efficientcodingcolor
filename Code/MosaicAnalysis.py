# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:03:56 2023
@author: David
"""

import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
from torch import nn
import pandas as pd
from numpy.linalg import norm
from analysis_utils import find_last_cp, find_last_model, reshape_flat_W, round_kernel_centers, scale, get_matrix, make_rr
from data import KyotoNaturalImages
from util import cycle, hexagonal_grid
from torch.utils.data import DataLoader
import math
from scipy.stats import pearsonr
import seaborn as sns
from sklearn.decomposition import PCA
import shapes
#from shapes import get_shape_module, Shape
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture
import scipy.optimize as opt
import scipy
from model import RetinaVAE, OutputTerms, OutputMetrics
#import cv2

#save = '240125-183448' #3 channels classic, parametrized
#save = '240225-003740' #1 channel classic, unparametrized

#save = '240210-215844' #M and S channels

save = '240301-055438'
#save2 = '240301-055438_test8'

path = "../../saves/" + save + "/" 
#path2 = "../../saves/" + save2 + "/" 

n_clusters_global = 4 #Best value for 240301-055438 is 4
n_comps_global = 3 #Best value for 240301-055438 is 3
rad_dist_global = 5 #Best value for 240301-055438 is 5
class Analysis():
        def __init__(self, path, epoch = None):
            self.interval = 1000
            self.path = path
            self.epoch = epoch
            if epoch == None:
                last_cp_file = find_last_cp(path)
                last_model_file = find_last_model(path)
            else:
                last_cp_file = "checkpoint-" + str(epoch) + ".pt"
                last_model_file = "model-" + str(epoch) + ".pt"
            
            self.max_iteration = int(last_cp_file[11:-3])
            self.iterations = np.array(range(self.interval,self.max_iteration,self.interval))
            self.cp = torch.load(path + last_cp_file)
            self.model = torch.load(path + last_model_file)
            
            if 'firing_restriction' in self.cp['args'].keys():
                self.firing_restriction = self.cp['args']['firing_restriction']
            else:
                self.firing_restriction = 'Lagrange'

            if 'shape' in self.cp['args'].keys():
                self.shape = self.cp['args']['shape']
            else: self.shape = None
            
            if self.shape is None:
                self.parametrized = False
            else:
                self.parametrized = True
            
            
            self.resp = None
            self.kernel_size = self.cp['args']['kernel_size']
            
            if 'n_colors' in self.cp['args'].keys():
                self.n_colors = self.cp['args']['n_colors']
            else:
                self.n_colors = 1
                
            self.n_neurons = self.cp['args']['neurons']
            
            if 'weights' in self.cp.keys():
                self.w_flat = self.cp['weights'].cpu().detach().numpy()
            else:
                self.w_flat = self.model.encoder.W.cpu().detach().numpy()
            self.w = reshape_flat_W(self.w_flat, self.n_neurons, self.kernel_size, self.n_colors)
            self.W = self.w
            
            center_surround_ratio = []
            for n in range(self.n_neurons):
                ON_sum = np.sum(np.clip(self.W[n,:,:,:],0,np.inf))
                OFF_sum = abs(np.sum(np.clip(self.W[n,:,:,:],-np.inf,0)))
                if np.max(self.W[n,:,:,:]) > abs(np.min(self.W[n,:,:,:])):
                    ratio = ON_sum/(ON_sum + OFF_sum)
                else:
                    ratio = OFF_sum/(ON_sum + OFF_sum)
                center_surround_ratio.append(ratio)
            self.center_surround_ratio = center_surround_ratio
            
            
            #self.L2_color = norm(self.w,axis = (1,2))
           

            
            lms_to_rgb = get_matrix(matrix_type = 'lms_to_rgb')
            self.lms_to_rgb = lms_to_rgb
            #self.W_rgb = np.tensordot(self.W, lms_to_rgb, axes = 1) #Gives red green opponency but thats a bug
            #self.W_rgb = np.transpose(np.tensordot(lms_to_rgb, np.transpose(self.W), axes = 1))

            #self.w_rgb = np.swapaxes(np.tensordot(lms_to_rgb, np.swapaxes(self.w, 0, 3), axes = 1), 0, 3)
            
            half_size = int(self.kernel_size/2)
            x = torch.arange(-half_size,half_size); y = torch.arange(-half_size,half_size)
            grid_x, grid_y = torch.meshgrid(x, y)
            rd = grid_x.flatten()**2 + grid_y.flatten()**2
            self.rd = torch.unsqueeze(rd,1)
            
            
        def get_DoG_params(self):
            if self.parametrized:
                self.a = self.model.encoder.shape_function.a.cpu().detach().numpy()
                self.b = self.model.encoder.shape_function.b.cpu().detach().numpy()
                self.c = self.model.encoder.shape_function.c.cpu().detach().numpy()
                self.d = self.model.encoder.shape_function.d.cpu().detach().numpy()
                self.max_d = np.argmax(abs(self.d), axis = 0)
                self.all_params = self.cp['model_state_dict']['encoder.shape_function.shape_params'].cpu().detach().numpy()
                self.fixed_centers = not 'encoder.kernel_centers' in self.cp['model_state_dict'].keys()
                
                if self.fixed_centers:
                    n_mosaics = self.cp['model_args']['n_mosaics']
                    self.kernel_centers = hexagonal_grid(self.n_neurons, self.kernel_size, n_mosaics).cpu().detach().numpy()
                else:
                    self.kernel_centers = self.cp['model_state_dict']['encoder.kernel_centers'].cpu().detach().numpy()
            
                    
                self.centers_round = np.clip(round_kernel_centers(self.kernel_centers), 0, self.kernel_size -1)
                self.RF_size = self.kernel_size
                self.RF_centers = self.kernel_centers 
            else:
                self.find_kernel_centers()
                self.RF_centers = self.kernel_centers
                self.RF_size = self.kernel_size
                
        def fit_Gaussian_2D(self, xy, x0, y0, amp_c, sigma_c):
            x,y = xy
            x0 = float(x0)
            y0 = float(y0)
            e = math.e
            center = amp_c*e**-(((x - x0)**2/sigma_c) + ((y-y0)**2/sigma_c))
            #surround = amp_s*e**-(((x - x0)**2/sigma_s) + ((y-y0)**2/sigma_s))
            #dog = center + surround 
            return center.ravel()

        def find_kernel_centers(self):
            n_neurons = self.W.shape[0]; kernel_size = self.W.shape[1]
            all_c = np.zeros([2, n_neurons])
            W = np.mean(abs(self.W), 3)
            
            
            initial_guess = (kernel_size/2,kernel_size/2,1,1)
            x = np.linspace(0,kernel_size-1,kernel_size)
            y = np.linspace(0,kernel_size-1,kernel_size)
            x, y = np.meshgrid(x,y)
            gauss_params = np.zeros([self.n_neurons, 4])
            for n in range(n_neurons):
                try:
                    params, cov = opt.curve_fit(self.fit_Gaussian_2D, (x,y), W[n,:,:].ravel(), p0 = initial_guess, maxfev=2000)
                    all_c[:,n] = params[0:2]
                    gauss_params[n,:] = params
                except RuntimeError: 
                    all_c[:,n] = [kernel_size/2, kernel_size/2]
            kernel_centers = np.flip(np.transpose(all_c),1)
            self.gauss_params = gauss_params
            self.kernel_centers = np.clip(kernel_centers, 0, np.inf)   
        
        
        def get_pathways(self, n_clusters): #Possible bug here please confirm it works
            best_bic = np.inf
            X = np.swapaxes(self.pca_transform,0,1)
            
            gauss = GaussianMixture(n_components = n_clusters, n_init = 200).fit(X)
            bic = gauss.bic(X)
                #if bic < best_bic:
                #    best_bic = bic
                #    best_gauss = gauss
            self.gauss = gauss
            type1 = gauss.predict(X)
            
            type_order = np.flip(np.argsort(np.bincount(type1)))
            type2 = []
            for i in range(type1.shape[0]):
                type2.append(list(type_order).index(type1[i]))
            
            self.type = np.transpose(type2)
            
        
        def make_mosaic(self, mosaic_type = None, ax = None, plot_size = False, save_fig = False, color = 'white'):
            if save_fig:
                matplotlib.use('agg')
                Videos_folder = '../Videos/' + save 
                mosaic_folder = Videos_folder + '/center_mosaic/'
                
                if not os.path.exists(Videos_folder):
                    os.mkdir(Videos_folder)
                if not os.path.exists(mosaic_folder):
                    os.mkdir(mosaic_folder)
            
            
            kernel_centers = self.kernel_centers
            if ax is None:
                fig, ax = plt.subplots(1,1)
        
        
            ax.set_aspect('equal')
            ax.set_facecolor('tab:gray')
            ax.set_xlim([0,self.kernel_size])
            ax.set_ylim([0,self.kernel_size])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            #ax.set_title("Mosaic type: " + str(mosaic_type), size = 20)
            for n in range(kernel_centers.shape[0]):
                marker = 'o'
                x = kernel_centers[n, 0]
                y = kernel_centers[n, 1]
                    
                if self.type[n] == mosaic_type or mosaic_type is None:
                    ax.plot(x, y, marker = marker, markersize = 12, color = color)

                    if plot_size and self.all_params is not None:
                        res_ratio = self.kernel_size/self.W.shape[1]
                        circle_center = plt.Circle((x,y), radius = self.zero_cross[n]*res_ratio, facecolor = 'none', edgecolor = 'r')
                        ax.add_patch(circle_center)
            if save_fig:
                plt.savefig(mosaic_folder + '/' + 'center_mosaic_' + str(int(self.epoch/self.interval)) + '.png')
                plt.close('all')
                    
        def mosaics(self, separate = True, plot_size = False):
            n_types = max(self.type) + 1
            colors =  ['black', 'blue', 'red', 'orange', 'green', 'purple', 'grey', 'cyan', 'teal']
            if separate:
                fig_len = math.ceil(np.sqrt(n_types))
            else:
                fig_len =1
            fig, axes = plt.subplots(fig_len, fig_len)
            if separate:
                axes = axes.flatten()
            for t in range(n_types):
                if separate:
                    axis = axes[t]
                else:
                    axis = axes
                self.make_mosaic(mosaic_type = t, ax = axis, plot_size = plot_size, color = colors[t])
            if separate:
                axes_remove = axes.shape[0] - n_types
                for i in range(axes_remove):
                    axes[-(i+1)].set_axis_off()
        
        def delete_mosaic(self, mosaic_num, save = False):
            to_keep = self.type != mosaic_num
            self.cp['model_state_dict']['encoder.shape_function.shape_params'] = self.cp['model_state_dict']['encoder.shape_function.shape_params'][:,to_keep]
            self.W = self.W[to_keep,:,:,:]
            self.model.encoder.shape_function.shape_params = nn.Parameter(self.model.encoder.shape_function.shape_params[:,to_keep])
            self.rad_avg = self.rad_avg[to_keep,:,:]
            self.type = self.type[to_keep]
            self.model.encoder.J = np.sum(to_keep)
            self.model.encoder.kernel_centers = nn.Parameter(self.model.encoder.kernel_centers[to_keep,:])
            self.model.encoder.logA = nn.Parameter(self.model.encoder.logA[to_keep])
            self.model.encoder.logB = nn.Parameter(self.model.encoder.logB[to_keep])
            self.model.Lambda = nn.Parameter(self.model.Lambda[to_keep])
            self.n_neurons = np.sum(to_keep)
            self.kernel_centers = self.kernel_centers[to_keep,:]
            if save:
                torch.save(self.model, path + "model-mosaic_deleted.pt")
                torch.save(self.cp, path + "checkpoint-mosaic_deleted.pt")
        #def make_mosaic_DoG_time(self, filename = 'mosaic_color', plot_size = False):
        #    matplotlib.use('agg')
        #    Videos_folder = '../Videos/' + save
        #    
        #    if not os.path.exists(Videos_folder):
        #        os.mkdir(Videos_folder)
        #    for i in range(self.iterations.shape[0]):
        #        if i%100 == 0:
        #            print(i)
        #        self.make_mosaic_DoG(i, 1, plot_size = plot_size)
        #        plt.savefig(Videos_folder + '/' + filename + '_' + str(i) + '.png')
        #        plt.close('all')
        #    matplotlib.use('qtagg')
        
        def make_kernels_image(self, weights, n_neurons = None, norm_each = True):
            if n_neurons == None:
                n_neurons = self.n_neurons
            sqrt_n = int(np.sqrt(n_neurons))
            if n_neurons == sqrt_n**2:
                add_row = 0
            else:
                add_row = 1
            kernel_size = weights.shape[1]
            kernels = np.zeros([kernel_size*sqrt_n, kernel_size*(sqrt_n+add_row), self.n_colors])
            for n in range(n_neurons):
                y_pos = (int(n/sqrt_n))
                x_pos = (n - y_pos*sqrt_n)
                x_pos = x_pos*kernel_size; y_pos = y_pos*kernel_size
                kernel_norm = norm(weights[n,:,:,:])
                for color in range(self.n_colors): 
                    this_kernel = weights[n,:,:,color]
                    if norm_each:
                        #this_kernel = this_kernel/norm(this_kernel)
                        this_kernel = this_kernel/kernel_norm
                    kernels[x_pos:x_pos+kernel_size,y_pos:y_pos+kernel_size, color] = this_kernel
            image = scale(kernels)
            plt.imshow(image)
            return image
        
        def return_neighbors(self, kernel_centers, check_same_pathway = True):
            n_neurons = kernel_centers.shape[0]
            c_nearest = np.zeros(n_neurons)
            for n1 in range(n_neurons):
                nearest = np.inf
                for n2 in range(n_neurons):
                    dist = ((kernel_centers[n2,0] - kernel_centers[n1,0])**2 + (kernel_centers[n2,1] - kernel_centers[n1,1])**2)**0.5
                    if check_same_pathway:
                        same_pathway = self.pathway[n1] == self.pathway[n2]
                    else:
                        same_pathway = True
                    if dist < nearest and n1 != n2 and same_pathway:
                        c_nearest[n1] = dist
                        nearest = dist
            nearest_min = np.sort(c_nearest)
            nearest_pairs = []
            for m in range(n_neurons):
                pair = [i for i, n in enumerate(c_nearest) if n == np.min(nearest_min[m])]
                nearest_pairs.append(pair)
            return nearest_pairs
        
        def plot3D(self, params = None, angles = None, size = 22, title = None, color_type = True, labels = 'PCA', ellipse = False):
            if params is None:
                params = self.pca_transform
            colors = ['black', 'blue', 'red', 'orange', 'green', 'purple', 'grey', 'cyan', 'teal']
            if angles is not None:
                elev, azim = angles
            else:
                elev, azim = (30,45)
            plt.figure()
            ax = plt.axes(projection='3d')
            if labels == 'LMS':
                ax.set_xlabel('L', size = size); ax.set_ylabel('M', size = size); ax.set_zlabel('S', size = size)
            elif labels == 'PCA':
                ax.set_xlabel('PC1', size = size); ax.set_ylabel('PC2', size = size); ax.set_zlabel('PC3', size = size)
            ax.view_init(elev = elev, azim = azim)
            
            if color_type:
                color = []
                for n in range(self.n_neurons):
                    color.append(colors[self.type[n]])
            else:
                color = 'black'
            ax.scatter(params[0,:], params[1,:], params[2,:], c = color)
            ax.set_title(title, size = 22)
            if ellipse:
                self.draw_ellipse(sig_ell = 0.2, ax = ax)
            
            
            return ax
            
        def plot3D_video(self, params, filename, title = None, size = 22, ellipse = False, special_command = None, color_type = True, labels = 'PCA'):
            azims = np.array(range(0,180,1))
            alts = np.array(range(30,90,1))
            alt = 30
            Videos_folder = '../Videos/' + save
            print(Videos_folder)
            matplotlib.use('agg')
            count = 0
            if not os.path.exists(Videos_folder):
                os.mkdir(Videos_folder)
            for azim in azims:
                ax = self.plot3D(params, angles = (alt, azim), title = title, color_type = color_type, labels = labels, ellipse = ellipse)
                plt.savefig('../Videos/' + save + '/' + filename + '_' + str(count) + '.png')
                count = count + 1
                plt.close()
                if special_command is not None:
                    exec(special_command)
            for alt in alts:
                ax = self.plot3D(params, angles = (alt, azim), title = title, color_type = color_type, labels = labels, ellipse = ellipse)
                if special_command is not None:
                    exec(special_command)
                plt.savefig('../Videos/' + save + '/' + filename + '_' + str(count) + '.png')
                count = count + 1
                plt.close()
                
        def get_images(self, restriction = 'True'):
            images = KyotoNaturalImages('kyoto_natim', self.kernel_size, True, 'cuda', self.n_colors, restriction)
            cov = images.covariance()
            self.images = images
            self.images_cov = cov
            return images, cov
        
            
        
        def get_responses(self, batch = 128, n_cycles = 100):
            images, cov = self.get_images()
            self.model.encoder.data_covariance = cov
            resp = []
            dets = []
            for i in range(n_cycles):
                images_load = next(cycle(DataLoader(images, batch))).to('cuda')
                images_sample = images_load.reshape([batch,self.n_colors,self.kernel_size**2])
                z, r, C_z, C_zx = self.model.encoder(image = images_sample, h_exp = 0, firing_restriction = 'Lagrange', corr_noise_sd = 0)
                
                L_numerator = C_z.cholesky()
                logdet_numerator = 2 * L_numerator.diagonal(dim1=-1, dim2=-2).log2().sum(dim=-1)
                L_denumerator = C_zx.cholesky()
                logdet_denuminator = 2 * L_denumerator.diagonal(dim1=-1, dim2=-2).log2().sum(dim=-1)
                det = logdet_numerator - logdet_denuminator
                dets.append(det.cpu().detach().numpy())
                
                
                resp.append(r.cpu().detach().numpy())
            self.resp = np.concatenate(resp, 0)
            self.cov_neurons = np.corrcoef(self.resp, rowvar = False)
            self.det = np.mean(dets)
            
        def compute_loss(self, batch = 128, n_cycles = 100, restriction = 'True', skip_read = False):
            if not skip_read:
                self.get_images(restriction) #BUG HERE restriction check will not always work. Can compute images with restriction then no restriction = don't recompute
            self.model.encoder.data_covariance = self.images_cov
            images = self.images
            losses, MI, r = [], [], []
            for this_cycle in range(n_cycles):
                images_load = next(cycle(DataLoader(images, batch))).to('cuda')
                images_sample = images_load.reshape([batch,self.n_colors,self.kernel_size**2])
                model = self.model(images_sample, h_exp = 0, firing_restriction = 'Lagrange', corr_noise_sd = 0)
                losses.append(model.calculate_metrics(self.epoch, 'Lagrange').final_loss('Lagrange').detach().cpu().item())
                MI.append(model.calculate_metrics(self.epoch, 'Lagrange').final_loss('None').detach().cpu().item())
                r.append(model.calculate_metrics(self.epoch, 'Lagrange').h.detach().cpu().mean().item())
                #loss =+ analysis.model(images_sample, h_exp = 0, firing_restriction = 'Lagrange', corr_noise_sd = 0).calculate_metrics(analysis.epoch, 'Lagrange').final_loss('Lagrange') #This is the line where forward gets called
                #MI_temp =+ analysis.model(images_sample, h_exp = 0, firing_restriction = 'Lagrange', corr_noise_sd = 0).calculate_metrics(analysis.epoch, 'Lagrange').final_loss('None') #This is the line where forward gets called
                del model, images_load, images_sample
            return losses,MI,r
            
            
        def get_cov_colors(self, batch = 100):
            images,cov = self.get_images()
            img_v1 = next(cycle(DataLoader(images, batch))).to('cuda')
            img_v2 = img_v1.reshape([batch,self.n_colors,self.kernel_size**2])

            self.model.encoder.data_covariance = cov

            z, r, C_z, C_zx = self.model.encoder(image = img_v2)
            r = r.cpu().detach().numpy()

            img_v3 = img_v2.detach().cpu().numpy()
            img_v4 = np.swapaxes(img_v3, 1, 2)
            img_v5 = np.concatenate(img_v4)
            img_v6 = np.swapaxes(img_v5, 0, 1)
            #X = np.reshape(np.concatenate(hey, 1), [3,1800*18])
            #Xt = np.swapaxes(X, 0, 1)
            self.cov_colors = np.cov(img_v6)
        
        def draw_ellipse(self, sig_ell=1, size = 22, tilt = 1, ax = None, alpha = 0.2):
            u1,s,v1 = np.linalg.svd(self.cov_colors)
            s = 1/np.sqrt(s)
            s_diag = np.zeros([self.n_colors,self.n_colors])
            np.fill_diagonal(s_diag, s)
            C = np.matmul(np.matmul(u1,s_diag), v1)
            n_samples = 100
            #angle1 = atan2(v[0,1],v[0,0])*360 / (2*np.pi)
            if ax is None:
                fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
                ax = fig.add_subplot(111, projection='3d')
            
            #coefs_pre = (np.sqrt(s[0])*sig_ell, np.sqrt(s[1])*sig_ell, np.sqrt(s[2]*sig_ell))  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
            # Radii corresponding to the coefficients:
            #coefs = coefs_pre# 1/np.sqr(coefs_pre)
            #coefs_rot = np.matmul(np.matmul(u1, coefs), v1) 
            #rx, ry, rz = coefs
            
            
            # Set of all spherical angles:
            u = np.linspace(0, 2 * np.pi, n_samples)
            v = np.linspace(0, np.pi, n_samples) 
            
            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            x = sig_ell * np.outer(np.cos(u + tilt), np.sin(v))
            y = sig_ell * np.outer(np.sin(u), np.sin(v))
            z = sig_ell * np.outer(np.ones_like(u), np.cos(v))
            x = x.reshape(n_samples**2); y = y.reshape(n_samples**2); z = z.reshape(n_samples**2)
            #ellipse = np.matmul(np.linalg.inv(C), np.array([x,y,z]))
            ellipse = np.matmul(np.linalg.inv(C), np.array([x,y,z]))
            # Plot:
            x = ellipse[0,:].reshape([n_samples, n_samples]); y = ellipse[1,:].reshape([n_samples, n_samples]); z = ellipse[2,:].reshape([n_samples, n_samples])
            ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha = alpha)
            
            # Adjustment of the axes, so that they all have the same span:
            max_radius = max(s)*2
            if ax is None:
                for axis in 'xyz':
                    getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))
                ax.set_xlabel('L', size = size); ax.set_ylabel('M', size = size); ax.set_zlabel('S', size = size)
                ax.set_zlim3d(-max_radius, max_radius)
                plt.show()
            
        def make_df(self):
            if hasattr(self, "type"):
                path = self.type
            else:
                path = np.repeat("None", self.n_neurons)                
            df_list = {'neuron': np.array(range(self.n_neurons)), 'center_x': self.kernel_centers[:,0], 'center_y':self.kernel_centers[:,1], 
                       'type':self.type, 'd.L':self.d[0,:], 'd.M':self.d[1,:], 'd.S':self.d[2,:], 'a.L':self.a[0,:], 'a.M':self.a[1,:], 'a.S':self.a[2,:], 
                       'center.L':self.center_colors[0,:], 'center.M':self.center_colors[1,:], 'center.S':self.center_colors[2,:]}
            self.df = pd.DataFrame(data = df_list)
        
        def make_df_pairs(self):
            pairs, neuron2_df, neuron1_type, neuron2_type, dist_df, corr_df = [], [], [], [], [], []
            neuron1_df = []
            i = 0
            p = 0.005
            for neuron1 in range(self.n_neurons):
                for neuron2 in range(self.n_neurons):
                    if np.random.binomial(1,p) == 1:
                        pair = [neuron1, neuron2] 
                        if pair not in pairs and list(np.flip(pair)) not in pairs:
                            i = i + 1
                            if i%100 == 0:
                                print(i)
                            pairs.append(pair)
                            neuron1_df.append(neuron1)
                            neuron2_df.append(neuron2)
                            neuron1_type.append(self.type[neuron1])
                            neuron2_type.append(self.type[neuron2])
                            dist_df.append(math.dist(self.kernel_centers[neuron1,:], self.kernel_centers[neuron2,:]))
                            if self.resp is not None:
                                corr = pearsonr(self.resp[:, neuron1], self.resp[:, neuron2])[0]
                                corr_df.append(corr)

            
            df_list = {'neuron1': neuron1_df, 'neuron2':neuron2_df, 'type1':neuron1_type, 'type2':neuron2_type, 'dist':dist_df, 'corr':corr_df}
            self.df_pairs = pd.DataFrame(data = df_list)
            self.df_pairs['same'] = self.df_pairs['type1'] == self.df_pairs['type2']
            
        def radial_averages(self, rad_range, high_res = True):
            all_y = np.around(self.RF_centers[:,0]).astype(int)
            all_x = np.around(self.RF_centers[:,1]).astype(int)
            rad_avg = np.zeros([self.n_neurons,rad_range,self.n_colors])
            self.rad_range = rad_range
            for n in range(self.n_neurons):
                y = all_y[n]
                x = all_x[n]
                if y >= self.RF_size:
                    y = self.RF_size - 1
                if x >= self.RF_size:
                    x = self.RF_size - 1
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                for r in range(rad_range):
                    tot = 4 
                    if y + r >= 0 and y + r <= self.RF_size - 1:
                        up = self.W[n, y + r, x, :]
                    else:
                        tot = tot - 1; up = 0
                    if y - r >= 0 and y - r <= self.RF_size - 1:  
                        down = self.W[n, y - r, x, :]
                    else:
                        tot = tot - 1; down = 0
                    if x + r >= 0 and x + r <= self.RF_size - 1:
                        right = self.W[n, y, x + r, :]
                    else:
                        tot = tot - 1; right = 0
                    if x - r >= 0 and x - r <= self.RF_size - 1:
                        left = self.W[n, y, x - r, :]
                    else:
                        tot = tot - 1; left = 0
                    if tot > 0:
                        rad_avg[n,r,:] = (up + down + right + left)/tot
                    else:
                        rad_avg[n,r,:] = 0
            self.rad_avg = rad_avg
            
        def zero_crossings_obs(self):
            zero_cross = np.zeros(self.n_neurons)
            for n in range(self.n_neurons):
                maxes = np.max(self.rad_avg[n,:,:], axis = 1)
                mins = np.min(self.rad_avg[n,:,:], axis = 1)
                signs = (maxes > abs(mins)).astype(int)
                crosses = np.roll(signs, -1) - (signs != 0).astype(int)
                first_cross = np.where(crosses)[0]
                zero_cross[n] = first_cross
        
        def zero_crossings(self):
            zero_cross = np.zeros(self.n_neurons)
            maxes = np.max(self.rad_avg, axis = 2)
            mins = np.min(self.rad_avg, axis = 2)
            signs = (maxes > abs(mins)).astype(int)
            crosses = np.roll(signs, -1, axis = 1) - (signs != 0).astype(int)
            for n in range(self.n_neurons):
                if crosses[n,:].any():
                    zero_cross[n] = np.where(crosses[n,:])[0][0] + 1
                else:
                    zero_cross[n] = self.rad_avg.shape[1] + 2
            self.zero_cross = zero_cross
                
                
                    
                    
                
        def plot_rads(self, type_num = None, title = "", hspace = 0.5):
            if type_num is not None:
                rad_avg = self.rad_avg[np.where(self.type == type_num)[0],:]
            else:
                rad_avg = self.rad_avg
            n_neurons = rad_avg.shape[0]
            size = round(np.sqrt(n_neurons))
            max_range = np.max(rad_avg)
            min_range = np.min(rad_avg)
            if size > int(np.sqrt(n_neurons)):
                extra_column = 0
            else:
                extra_column = int((n_neurons - size**2)/size - 0.0001) + 1
            fig, axes = plt.subplots(size,size+extra_column)
            plt.subplots_adjust(hspace = hspace)
            for n in range(size*(size+extra_column)):
                y_pos = int(n/size)
                x_pos = n - y_pos*size,
                axis = axes[x_pos, y_pos][0]
                if n < n_neurons:
                    
                    self.plot_rad_avg(rad_avg[n,:,:], axis)
                    #axis = return_subplot(axes, n, n_neurons)
                    #axis.plot(rad_avg[n,:,0], color = 'r')
                    #axis.plot(rad_avg[n,:,1], color = 'g')
                    #axis.plot(rad_avg[n,:,2], color = 'b')
                    #axis.axhline(0, color = 'black')
                    axis.set_title("# " + str(n), fontsize = 6)
                    if n != 0:
                        xax, yax = axis.get_xaxis(), axis.get_yaxis()
                        xax.set_visible(False); yax.set_visible(False)
                    axis.set_ylim(min_range, max_range)
                else:
                    axis.set_visible(False)
            fig.suptitle(title, y = 0.95, size = 30)
            fig.supxlabel('Radial distance from center (pixels)', y = 0.05, size = 24)
            fig.supylabel('Weight', x = 0.07, size = 24)
        
        def plot_rad_avg(self, rad_avg, axis = None, labels = False):
            if axis is None:
                fig, axis = plt.subplots(1,1)
            if self.n_colors == 3:
                colors = ['r','g','b']
            else:
                colors = ['r', 'b']
            for c in range(self.n_colors):
                axis.plot(rad_avg[:,c], color = colors[c])
            axis.axhline(0, color = 'black')
            if labels:
                axis.set_xlabel("Distance from center", size = 50)
                axis.set_ylabel("Weight", size = 50)
        
        #Plots one RF with its radial average 
        def plot_RF_rad(self,n):
            fig, axes = plt.subplots(1,2)
            axes[0].imshow(scale(self.W[n,:,:,:]))
            self.plot_rad_avg(self.rad_avg[n,:,:], axes[1], labels = True)
            fig.suptitle(save + " neuron# " + str(n))
            
            
        
        def pca_radial_average(self, n_comp, plot = True):
            colors = ['black', 'blue', 'orange', 'red', 'yellow']
            if not hasattr(self, 'rad_avg'):
                self.radial_averages()
            rad_avg_con = np.empty([self.n_neurons, 0])
            for c in range(self.n_colors):
                rad_avg_con = np.concatenate([rad_avg_con, self.rad_avg[:,:,c]], axis = 1)
            pca = PCA(n_components=n_comp)
            pca.fit(rad_avg_con)
            self.pca = pca
            self.pca_transform = np.swapaxes(pca.transform(rad_avg_con),0,1)
            if plot:
                plt.figure()
                patches = []
                line_length = int(self.pca.components_.shape[1]/self.n_colors)
                for comp in range(n_comp):
                    for color in range(self.n_colors):
                        x_range = np.arange(line_length*color, line_length*(color+1))
                        plt.plot(x_range, self.pca.components_[comp,x_range], color = colors[comp], linewidth = 5)
                        plt.plot(x_range, self.pca.components_[comp,x_range], 'o', color = colors[comp], markersize = 10)
                    patches.append(mpatches.Patch(color=colors[comp], label='PC' + str(comp + 1) + ": " + str(format(pca.explained_variance_ratio_[comp] * 100, '.1f')) + " %", linewidth=12, alpha=0.5))
                plt.xlabel('Distance from center', size = 40)
                plt.ylabel('PCA loadings', size = 40)
                plt.title("First " + str(n_comp) + " PCAs of radial RFs", size = 22)
                
                    
                plt.axhline(y=0, xmin = 0, xmax = self.pca.components_.shape[1])
                plt.axvline(x=self.rad_range, color = 'black')
                plt.axvline(x=self.rad_range*2, color = 'black')
                plt.xticks(ticks = [int(self.rad_range/2), int(self.rad_range*1.5), int(self.rad_range*2.5)], labels = ['Long cones', 'Medium cones', 'Short cones'], size = 40)
                plt.yticks([])
                
                plt.legend(handles=patches, loc='upper right', framealpha=0.5, frameon=True, fontsize = 30)
        
        def increase_res(self, new_size, norm_size = False):
            if new_size%2 !=0:
                Exception("New size must be even!")
            step = self.kernel_size/new_size
            x = torch.arange(0, self.kernel_size, step)
            y = torch.arange(0, self.kernel_size, step)
            kernel_x = torch.tensor(self.kernel_centers[:, 0])
            kernel_y = torch.tensor(self.kernel_centers[:, 1])

            grid_x, grid_y = torch.meshgrid(x, y)
            grid_x = grid_x.flatten().float()
            grid_y = grid_y.flatten().float()
            dx = kernel_x[None, :] - grid_x[:, None]
            dy = kernel_y[None, :] - grid_y[:, None]
            if norm_size and hasattr(self, 'a'):
                size_norm = np.mean(self.a*abs(self.d), axis = 0)
            else:
                size_norm = 1
            dist = (dx**2 + dy**2)/size_norm
            
            dog = shapes.DifferenceOfGaussianShape(kernel_size = new_size, n_colors = self.n_colors, num_shapes = self.n_neurons, init_params = self.all_params, read= True).shape_function(dist)
            dog = dog.reshape([self.n_colors, new_size,new_size,self.n_neurons])
            dog = np.swapaxes(dog.detach().cpu().numpy(),0,3)
            dog_norm = np.zeros(dog.shape)
            for n in range(self.n_neurons):
                dog_norm[n,:,:,:] = (dog[n,:,:,:]/norm(dog[n,:,:,:]))/(step)
            self.W = dog_norm
            self.RF_size = new_size
            self.RF_centers = self.kernel_centers * (self.RF_size/self.kernel_size)
            self.centers_round = np.clip(round_kernel_centers(self.RF_centers), 0, self.RF_size -1)
            self.center_colors = self.W[:,self.centers_round[:,0], self.centers_round[:,1],:]
            
            
        def make_RF_from_pca(self):
            comps = self.pca.components_
            n_comps = comps.shape[0]
            
            if comps.shape[1]%self.n_colors == 0:
                size = int(comps.shape[1]/self.n_colors)
            else:
                Exception("Incompatible shapes of pca comps!")
            comps = np.reshape(comps, [n_comps, self.n_colors, size])
            comps = np.swapaxes(comps, 1, 2)
                
            lin = np.linspace(-size,size,(size*2)+1, dtype = int)
            x_grid,y_grid = np.meshgrid(lin,lin)
            RF_pca = np.zeros([n_comps, x_grid.shape[0], x_grid.shape[1], self.n_colors])
            for comp in range(n_comps):
                for color in range(self.n_colors):
                    for x in range(lin.shape[0]):
                        for y in range(lin.shape[0]):
                            x_dist = lin[x]
                            y_dist = lin[y]
                            dist = round(np.sqrt(x_dist**2 + y_dist**2))
                            if dist >= size:
                                dist = 14
                            RF_pca[comp, x , y, color] = comps[comp, dist, color]
            self.RF_pca = RF_pca
            
        def fit_DoG(self, device = "cuda:0", LR = 0.001, n_steps = 20000):
            all_loss = []
            kernel_centers = nn.Parameter(torch.tensor(self.kernel_centers, device = device))
            DoG_mod = shapes.get_shape_module("difference-of-gaussian")(torch.tensor(self.kernel_size, device = device), self.n_colors, torch.tensor(self.n_neurons, device = device)).to(device)
            params = DoG_mod.shape_params
            RFs = torch.tensor(self.w_flat, device = device)
            optimizer = torch.optim.SGD([kernel_centers, DoG_mod.shape_params], lr = LR)
            for i in range(n_steps):
                optimizer.zero_grad()
                RFs_DoG = DoG_mod(kernel_centers)
                loss = torch.sum((RFs - RFs_DoG)**2)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if i%100 == 0:
                    if i%1000 == 0:
                        print(loss, i)
                    all_loss.append(loss.detach().cpu().numpy())
            RFs_fit = np.swapaxes(np.reshape(RFs_DoG.detach().cpu().numpy(), [self.n_colors,self.kernel_size,self.kernel_size,self.n_neurons]), 0, 3)
            self.RFs_fit = RFs_fit
            self.DoG_mod = DoG_mod
            self.a, self.b, self.c, self.d = DoG_mod.a.cpu().numpy(), DoG_mod.b.cpu().numpy(), DoG_mod.c.cpu().numpy(), DoG_mod.d.cpu().numpy()
            self.max_d = np.argmax(abs(self.d), axis = 0)
            self.all_params = params.detach().cpu().numpy()
            self.kernel_centers = kernel_centers.detach().cpu().numpy()
            self.DoG_fit_losses = all_loss
            
            r_coefs = []
            for i in range(self.n_neurons):
              fit_flat = RFs_fit[i,:,:,:].flatten()
              og_flat = self.w[i,:,:,:].flatten()
              coef = np.corrcoef(og_flat, fit_flat)
              r_coefs.append(coef[1,0])  
             
            self.DoG_r = np.array(r_coefs)
            
        def compare_DoG_fits(self, n):
            fig, axes = plt.subplots(1,2)
            axes[0].imshow(scale(self.w[n,:,:,:]))
            axes[0].set_title("Unparametrized RF", size = 30)
            axes[1].imshow(scale(self.RFs_fit[n,:,:,:]))
            axes[1].set_title("DoG fit", size = 30)
            plt.suptitle("cor = " + str(round(self.DoG_r[n],4)) + ", " + save + " #" + str(n), size = 30)
        
        def DoG_fit_func(self, shape, centers):
            def W_from_shapes(params):
                shape.shape_params = nn.Parameter(torch.tensor(params, device = "cuda:0"), requires_grad = False)
                fit = shape(torch.tensor(centers, device = "cuda:0")).detach().cpu().numpy()
                return np.sum(self.w_flat - fit)**2
            return W_from_shapes
            
            #Doesn't work right now :(
        def fit_DoG_scipy(self, device = "cuda:0"):
            DoG_mod = shapes.get_shape_module("difference-of-gaussian")(torch.tensor(self.kernel_size, device = device), self.n_colors, torch.tensor(self.n_neurons, device = device)).to(device)
            init_params = DoG_mod.shape_params
            fun = self.DoG_fit_func(DoG_mod, self.kernel_centers)
            optimization = scipy.optimize.minimize(fun, init_params.detach().cpu().numpy(), method = "Nelder-Mead")
            params = np.swapaxes(optimization['x'].reshape([self.n_neurons, 4*self.n_colors]),0,1)
            
            DoG_mod.shape_params = nn.Parameter(torch.tensor(params, device = device), requires_grad = False)
            RFs_DoG = DoG_mod(torch.tensor(self.kernel_centers, device = device)).detach().cpu().numpy()
            
            RFs_fit = np.swapaxes(np.reshape(RFs_DoG, [self.n_colors,self.kernel_size,self.kernel_size,self.n_neurons]), 0, 3)
            self.RFs_fit = RFs_fit
            self.DoG_mod = DoG_mod
            self.a, self.b, self.c, self.d = DoG_mod.a.cpu().numpy(), DoG_mod.b.cpu().numpy(), DoG_mod.c.cpu().numpy(), DoG_mod.d.cpu().numpy()
            self.max_d = np.argmax(abs(self.d), axis = 0)
            self.all_params = params
            
            r_coefs = []
            for i in range(self.n_neurons):
              fit_flat = RFs_fit[i,:,:,:].flatten()
              og_flat = self.w[i,:,:,:].flatten()
              coef = np.corrcoef(og_flat, fit_flat)
              r_coefs.append(coef[1,0])  
             
            self.DoG_r = r_coefs
        
        
        #Currently only works for 2 channels 
        def change_d(self, cluster, d, save = True):
            params = self.model.encoder.shape_function.shape_params.detach()
            W = self.cp['weights'].detach()
            rad_avg = self.rad_avg
            half_W = int(W.shape[0]/len(d))
            
            for n in range(self.n_neurons):
                if self.type[n] == cluster:
                    params[3,n] *= d[0]
                    params[7,n] *= d[1]
                    
                    rad_avg[n,:,0] *= d[0]
                    rad_avg[n,:,1] *= d[1]
                    
                    W[:half_W,n] *= d[0]
                    W[half_W:,n] *= d[1]
                    
                
            params.requires_grad = True
            self.model.encoder.shape_function.shape_params = nn.Parameter(params)
            self.cp['model_state_dict']['encoder.shape_function.shape_params'] = nn.Parameter(params)
            self.cp['weights'] = nn.Parameter(W)
            self.rad_avg = rad_avg
            
            if save:
                torch.save(self.model, path + "model-newD.pt")
                torch.save(self.cp, path + "checkpoint-newD.pt")
            
        
        def __call__(self, n_comps = None, rad_dist = None, n_clusters = None):
            if n_comps is None:
                n_comps = n_comps_global
            if rad_dist is None:
                rad_dist = rad_dist_global
            if n_clusters is None:
                n_clusters = n_clusters_global
            plt.close('all')
            self.get_DoG_params()
            if self.parametrized:
                self.radial_averages(rad_dist)
                self.pca_radial_average(n_comp = n_comps, plot = False)
                self.get_pathways(n_clusters)
            else:
                self.fit_DoG()
                #self.fit_DoG_scipy()
                self.radial_averages(rad_dist)
                self.pca_radial_average(n_comp = n_comps, plot = False)
                #self.make_RF_from_pca()
                self.get_pathways(n_clusters)
                self.zero_crossings()
            matplotlib.use("QtAgg")
                
            
            #self.neighbors = self.return_neighbors(self.kernel_centers, check_same_pathway = True)
            #self.get_responses()
            #self.make_df()
            #self.make_df_pairs()
            #self.get_cov_colors()
            #self.images.pca_color()
            #matplotlib.use("Qtagg")

class Analysis_time():
    def __init__(self, path, interval, n_comps, rad_dist, n_clusters, start_epoch = 0, stop_epoch = np.inf):
        self.interval = interval
        last_cp_file = find_last_cp(path)
        last_model_file = find_last_model(path) 
        self.max_iteration = int(last_cp_file[11:-3])
        self.iterations = np.array(range(start_epoch + self.interval,self.max_iteration,self.interval))
        self.n_analyses = int(self.max_iteration/self.interval)
        
        all_analyses = []

        for iteration in self.iterations:
            if iteration >= start_epoch and iteration <= stop_epoch:
                try:
                    all_analyses.append(Analysis(path, epoch = iteration))
                except FileNotFoundError:
                    print("Skipped epoch " + str(iteration) + " because file could not be found")
                    
                if iteration%100000 == 0:
                    print(iteration)
        self.analyses = all_analyses
        self.last = self.analyses[-1]
        self.last(n_comps, rad_dist, n_clusters)
        self.rad_dist = rad_dist
        self.n_clusters = n_clusters
        self.n_comps = n_comps
        
    def mosaic_type_video(self, filename, separate, ref = -1):
        matplotlib.use('Qtagg')
        Videos_folder = '../Videos/' + save
        self.analyses[ref](self.n_comps, self.rad_dist, self.n_clusters)
        n_type = self.analyses[ref].type
        n = 0
        for analysis in self.analyses:
            n += 1
            analysis.type = n_type
            analysis.get_DoG_params()
            analysis.mosaics(separate)
            manager = plt.get_current_fig_manager()
            manager.full_screen_toggle()
            plt.savefig(Videos_folder + '/' + filename + '_' + str(n) + '.png')
            plt.close('all')
            if n%10 == 0:
                print(n, ' center mosaic')
        matplotlib.use('Qtagg')
    def epoch_metrics(self, batch = 128, n_cycles = 50):
        det_nums = []
        det_denums = []
        i = 0
        a = np.empty([self.last.n_colors, self.last.n_neurons, 0])
        b = np.empty([self.last.n_colors, self.last.n_neurons, 0])
        c = np.empty([self.last.n_colors, self.last.n_neurons, 0])
        d = np.empty([self.last.n_colors, self.last.n_neurons, 0])
        losses = []
        MI = []
        r = []
        images, cov = self.last.get_images()
        for analysis in self.analyses:
            #num = analysis.model.encoder.C_z.detach().cpu().numpy()
            #denum = analysis.model.encoder.C_zx.detach().cpu().numpy()
            #det_nums.append(np.linalg.det(num))
            #det_denums.append(np.linalg.det(denum))
            if n_cycles > 0:
                analysis.images_cov = cov
                analysis.images = images
                loss, MI_temp, r_temp = analysis.compute_loss(batch = batch, n_cycles = n_cycles, skip_read = True)
                losses.append(np.mean(loss.detach().cpu().item()))
                MI.append(np.mean(MI_temp.detach().cpu().item()))
                r.append(np.sum(r_temp.detach().cpu().item()))
                if i%1 == 0:
                    print(i)
            #all_a = self.analyses[i].model.encoder.shape_function.a.cpu().detach().numpy()
            a = np.append(a, np.expand_dims(self.analyses[i].model.encoder.shape_function.a.cpu().detach().numpy(), 2), axis = 2)
            b = np.append(b, np.expand_dims(self.analyses[i].model.encoder.shape_function.b.cpu().detach().numpy(), 2), axis = 2)
            c = np.append(c, np.expand_dims(self.analyses[i].model.encoder.shape_function.c.cpu().detach().numpy(), 2), axis = 2)
            d = np.append(d, np.expand_dims(self.analyses[i].model.encoder.shape_function.d.cpu().detach().numpy(), 2), axis = 2)
            i += 1
        
        self.det_nums = det_nums
        self.det_denums = det_denums
        self.MI = MI
        self.losses = losses
        self.r = r
        
        self.a, self.b, self.c, self.d = a,b,c,d
    
    def plot_params_time(self, n):
        n_colors = self.last.n_colors
        n_params = 4
        params_time = [self.a, self.b, self.c, self.d]
        params_time_str = ['Center precision','Surround precision','Surround strength','Color strength']
        colors_str = ['r', 'g', 'b']
        fig, axes = plt.subplots(1,n_params)
        for param in range(n_params):
            param_time = params_time[param]
            for color in range(n_colors):
                axes[param].plot(param_time[color,n,:], color = colors_str[color])
            axes[param].set_title(params_time_str[param], size = 20)
        fig.suptitle("Neuron # " + str(n) + ', type: ' + str(self.last.type[n]), size = 30)
        fig.supxlabel("Epoch", size = 30)
            
                
    #eg: "center_mosaic.mp4" or "center_mosaic.avi"
#def make_video(video_name):
#    image_folder = '../Videos/' + save + '/' + video_name + '/'
#    video_name = video_name + '.mp4'
#    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
#    print('hello')
#    frame = cv2.imread(os.path.join(image_folder, images[0]))
#    height, width, layers = frame.shape
#    
#    video = cv2.VideoWriter(video_name, 0, 1, (width,height))
#    print('hello')
#    a = True
#    for image in images:
#        #print(image)
#        hello = cv2.imread(os.path.join(image_folder, image))
#        if a:
#            print('hey')
#            a = False
#        video.write(cv2.imread(os.path.join(image_folder, image)))
#    
#    cv2.destroyAllWindows()
#    video.release()
            
        
    

test = Analysis(path, 2000000)
#test2 = Analysis(path2)
test(n_comps_global, rad_dist_global, n_clusters_global)#, test2(2)
#test2(n_comps_global, rad_dist_global, 5)
#test_all = Analysis_time(path, 5000, n_comps_global, rad_dist_global, n_clusters_global)#, stop_epoch = 3000000)
#test_all.epoch_metrics()