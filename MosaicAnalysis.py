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
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.mixture import GaussianMixture


#save = '230625-235533' #Loos nice, 300 neurons
#save = '230705-141246' #500 neurons 12x12x3
#save = '230813-224700' #500 neurons 12x12x3 but [50,30,20] color pca split. David's old method
#save = '230821-024538' #500 neurons 12x12x3 but [50,30,20] color pca split. John's method 
save = '230825-143401' #Fixed kernel centers
#save = '230828-152654' #500 neurons x12x12x3 but [80,15,5] pca split
path = "saves/" + save + "/"

class Analysis():
        def __init__(self, path, DoG = True):
            self.interval = 1000
            self.path = path
            last_cp_file = find_last_cp(path)
            last_model_file = find_last_model(path)
            self.max_iteration = int(last_cp_file[11:-3])
            self.iterations = np.array(range(self.interval,self.max_iteration,self.interval))
            self.cp = torch.load(path + last_cp_file)
            self.model= torch.load(path + last_model_file)
            self.a = self.model.encoder.shape_function.a.cpu().detach().numpy()
            self.b = self.model.encoder.shape_function.b.cpu().detach().numpy()
            self.c = self.model.encoder.shape_function.c.cpu().detach().numpy()
            self.d = self.model.encoder.shape_function.d.cpu().detach().numpy()
            self.resp = None
            self.fixed_centers = not 'encoder.kernel_centers' in self.cp['model_state_dict'].keys()
            
            self.all_params = self.cp['model_state_dict']['encoder.shape_function.shape_params'].cpu().detach().numpy()
            self.kernel_size = self.cp['args']['kernel_size']
            self.n_colors = self.cp['args']['n_colors']
            self.n_neurons = self.cp['args']['neurons']
            self.w_flat = self.cp['weights'].cpu().detach().numpy()
            self.w = reshape_flat_W(self.w_flat, self.n_neurons, self.kernel_size, self.n_colors)
            self.W = self.w
            
            if not self.fixed_centers:
                self.kernel_centers = self.cp['model_state_dict']['encoder.kernel_centers'].cpu().detach().numpy()
            else:
                n_mosaics = self.cp['model_args']['n_mosaics']
                self.kernel_centers = hexagonal_grid(self.n_neurons, self.kernel_size, n_mosaics).cpu().detach().numpy()
            
            self.L2_color = norm(self.w,axis = (1,2))
            self.centers_round = np.clip(round_kernel_centers(self.kernel_centers), 0, self.kernel_size -1)
            self.RF_size = self.kernel_size
            self.RF_centers = self.kernel_centers
            
            lms_to_rgb = get_matrix(matrix_type = 'lms_to_rgb')
            self.lms_to_rgb = lms_to_rgb
            #self.W_rgb = np.tensordot(self.W, lms_to_rgb, axes = 1) #Gives red green opponency but thats a bug
            #self.W_rgb = np.transpose(np.tensordot(lms_to_rgb, np.transpose(self.W), axes = 1))
            self.w_rgb = np.swapaxes(np.tensordot(lms_to_rgb, np.swapaxes(self.w, 0, 3), axes = 1), 0, 3)
            
            half_size = int(self.kernel_size/2)
            x = torch.arange(-half_size,half_size); y = torch.arange(-half_size,half_size)
            grid_x, grid_y = torch.meshgrid(x, y)
            rd = grid_x.flatten()**2 + grid_y.flatten()**2
            self.rd = torch.unsqueeze(rd,1)
        
        def get_params_time(self):
            i = -1; first = True
            iterations = self.iterations
            for iteration in iterations:
                i = i + 1
                cp = torch.load(path + 'checkpoint-' + str(iteration) + '.pt')
                model = torch.load(path + 'model-' + str(iteration) + '.pt')
                all_params = cp['model_state_dict']['encoder.shape_function.shape_params'].cpu().detach().numpy()
                if not self.fixed_centers:
                    kernel_centers = cp['model_state_dict']['encoder.kernel_centers'].cpu().detach().numpy()
                else: kernel_centers = self.kernel_centers
                    
                weights_flat = cp['weights'].cpu().detach().numpy()
                weights = reshape_flat_W(weights_flat, self.n_neurons, self.kernel_size, self.n_colors)
                
                a = model.encoder.shape_function.a.cpu().detach().numpy()
                b = model.encoder.shape_function.b.cpu().detach().numpy()
                c = model.encoder.shape_function.c.cpu().detach().numpy()
                d = model.encoder.shape_function.d.cpu().detach().numpy()
                if first:
                    all_params_time = np.zeros([iterations.shape[0], all_params.shape[0], all_params.shape[1]])
                    kernel_centers_time = np.zeros([iterations.shape[0], kernel_centers.shape[0], kernel_centers.shape[1]])
                    weights_time = np.zeros([iterations.shape[0], weights.shape[0], weights.shape[1], weights.shape[2], weights.shape[3]])
                    a_time = np.zeros([iterations.shape[0], test.a.shape[0], test.a.shape[1]])
                    b_time = np.zeros([iterations.shape[0], test.b.shape[0], test.b.shape[1]])
                    c_time = np.zeros([iterations.shape[0], test.c.shape[0], test.c.shape[1]])
                    d_time = np.zeros([iterations.shape[0], test.d.shape[0], test.d.shape[1]])
                    first = False
                all_params_time[i,:,:] = all_params
                kernel_centers_time[i,:,:] = kernel_centers
                weights_time[i,:,:,:,:] = weights
                
                a_time[i,:,:] = a
                b_time[i,:,:] = b
                c_time[i,:,:] = c
                d_time[i,:,:] = d
            
            self.all_params_time = all_params_time; self.kernel_centers_time = kernel_centers_time; self.W_time = weights_time
            self.a_time = a_time; self.b_time = b_time; self.c_time = c_time; self.d_time = d_time
        
        def get_pathways(self): #Possible bug here please confirm it works
            best_bic = np.inf
            X = np.swapaxes(self.pca_transform,0,1)
            
            gauss = GaussianMixture(n_components = 7, n_init = 50).fit(X)
            bic = gauss.bic(X)
                #if bic < best_bic:
                #    best_bic = bic
                #    best_gauss = gauss
            self.gauss = gauss
            self.type = gauss.predict(X)
            
        
        def make_mosaic_DoG(self, t = 'last', n_plots = 1, plot_size = False):
            if t == 'last':
                t = self.iterations.shape[0] - 1
            if n_plots > self.n_colors:
                Exception("You can't have more plots than color channels!")
                
            kernel_centers = self.kernel_centers
            fig, axes = plt.subplots(1,n_plots)
            if n_plots == 1:
                axes = [axes]
            color = -1
            if n_plots == 3:
                lms_str = ['L cones', 'M cones', 'S cones']
            else:
                lms_str = ['RF centers']
            
            for ax in axes:
                ax.set_aspect('equal')
                ax.set_facecolor('tab:gray')
                color = color + 1
                ax.set_xlim([0,self.kernel_size])
                ax.set_ylim([0,self.kernel_size])
                ax.set_title('Mosaic of ' + lms_str[color], size = 12)
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                for n in range(self.n_neurons):
                    #if self.pathway[n] == 'ON':
                    #    marker = 'x'
                    #elif self.pathway[n] == 'OFF':
                    #    marker = 'o'
                    #else:
                    #    marker = '.'
                    marker = 'o'
                    x = kernel_centers[n, 0]
                    y = kernel_centers[n, 1]
                    #if self.type[n] == 'white' or self.type[n] == 'black':
                    ax.plot(x, y, marker = marker, markersize = 6, color = 'white')

                    if plot_size and self.all_params is not None:
                        
                        circle_center = plt.Circle((x,y), radius = self.a, facecolor = 'none', edgecolor = 'r')
                        circle_surround = plt.Circle((x,y), radius = self.b, facecolor = 'none', edgecolor = 'b')
                        ax.add_patch(circle_center)
                        ax.add_patch(circle_surround)
        
        def make_mosaic_DoG_time(self, filename = 'mosaic_color', plot_size = False):
            matplotlib.use('agg')
            Videos_folder = 'Videos/' + save
            if not os.path.exists(Videos_folder):
                os.mkdir(Videos_folder)
            for i in range(self.iterations.shape[0]):
                if i%100 == 0:
                    print(i)
                self.make_mosaic_DoG(i, 1, plot_size = plot_size)
                plt.savefig('Videos/' + save + '/' + filename + '_' + str(i) + '.png')
                plt.close('all')
            matplotlib.use('qtagg')
        
        def make_kernels_image(self, weights, norm_each = True, n_neurons = None):
            if n_neurons == None:
                n_neurons = self.n_neurons
            sqrt_n = int(np.sqrt(n_neurons))
            if n_neurons == sqrt_n**2:
                add_row = 0
            else:
                add_row = 1
            kernels = np.zeros([self.kernel_size*sqrt_n, self.kernel_size*(sqrt_n+add_row), self.n_colors])
            for n in range(n_neurons):
                y_pos = (int(n/sqrt_n))
                x_pos = (n - y_pos*sqrt_n)
                x_pos = x_pos*self.kernel_size; y_pos = y_pos*self.kernel_size
                kernel_norm = norm(weights[n,:,:,:])
                for color in range(self.n_colors): 
                    this_kernel = weights[n,:,:,color]
                    if norm_each:
                        #this_kernel = this_kernel/norm(this_kernel)
                        this_kernel = this_kernel/kernel_norm
                    kernels[x_pos:x_pos+self.kernel_size,y_pos:y_pos+self.kernel_size, color] = this_kernel
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
        
        def plot3D(self, params, angles = None, size = 22, title = None, color_type = False, labels = 'LMS', ellipse = False):
            colors = ['black', 'blue', 'red', 'yellow', 'green', 'orange', 'grey']
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
            
        def plot3D_video(self, params, filename, title = None, size = 22, ellipse = False, special_command = None, color_type = True, labels = 'LMS'):
            azims = np.array(range(0,180,1))
            alts = np.array(range(30,90,1))
            alt = 30
            Videos_folder = 'Videos/' + save
            matplotlib.use('agg')
            count = 0
            if not os.path.exists(Videos_folder):
                os.mkdir(Videos_folder)
            for azim in azims:
                ax = self.plot3D(params, angles = (alt, azim), title = title, color_type = color_type, labels = labels, ellipse = ellipse)
                plt.savefig('Videos/' + save + '/' + filename + '_' + str(count) + '.png')
                count = count + 1
                plt.close()
                if special_command is not None:
                    exec(special_command)
            for alt in alts:
                ax = self.plot3D(params, angles = (alt, azim), title = title, color_type = color_type, labels = labels, ellipse = ellipse)
                if special_command is not None:
                    exec(special_command)
                plt.savefig('Videos/' + save + '/' + filename + '_' + str(count) + '.png')
                count = count + 1
                plt.close()
                
        def get_images(self):
            images = KyotoNaturalImages('kyoto_natim', self.kernel_size, True, 'cuda', self.n_colors)
            cov = images.covariance()
            self.images = images
            return images, cov
        
        def get_responses(self, batch = 100, n_cycles = 1000):
            images, cov = self.get_images()
            self.model.encoder.data_covariance = cov
            resp = []
            for i in range(n_cycles):
                images_load = next(cycle(DataLoader(images, batch))).to('cuda')
                images = images_load.reshape([batch,self.n_colors,self.kernel_size**2])
                z, r, C_z, C_zx = self.model.encoder(image = images)
                resp.append(r.cpu().detach().numpy())
            self.resp = np.concatenate(resp, 0)
            self.cov_neurons = np.corrcoef(self.resp, rowvar = False)
        
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
            
        def radial_averages(self, rad_range = None, high_res = True):
            if rad_range is None:
                rad_range = int(self.RF_size/2)

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
            
        def plot_radial_averages(self, rad_avg, title = "", hspace = 0.5):
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
                    #axis = return_subplot(axes, n, n_neurons)
                    axis.plot(rad_avg[n,:,0], color = 'r')
                    axis.plot(rad_avg[n,:,1], color = 'g')
                    axis.plot(rad_avg[n,:,2], color = 'b')
                    axis.axhline(0, color = 'black')
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
            
        def pca_radial_average(self, n_comp = 4, plot = True):
            colors = ['black', 'blue', 'orange', 'red', 'yellow']
            if not hasattr(self, 'rad_avg'):
                self.radial_averages()
            rad_avg_con = np.concatenate(([self.rad_avg[:,:,0], self.rad_avg[:,:,1], self.rad_avg[:,:,2]]), axis = 1)
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
            if norm_size:
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
            self.center_colors = self.W[:,self.centers_round[:,0], self.centers_round[:,1]]
            
            
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
            
        
        def __call__(self):
            plt.close('all')
            self.get_params_time()
            self.increase_res(100, norm_size = True)
            
            self.kernels_image = self.make_kernels_image(self.w, n_neurons = 50)
            self.radial_averages(15)
            self.pca_radial_average(plot = False)
            self.make_RF_from_pca()
            self.get_pathways()
            
            #self.neighbors = self.return_neighbors(self.kernel_centers, check_same_pathway = True)
            self.get_responses()
            #self.make_df()
            #self.make_df_pairs()
            #self.get_cov_colors()
            self.images.pca_color()
            matplotlib.use("Qtagg")
            
test = Analysis(path)
test()