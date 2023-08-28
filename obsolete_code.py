# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:30:54 2023

@author: Rache
"""

        def get_pathways(self): #Possible bug here please confirm it works
            W_sum = np.mean(self.W, axis = 3)
            centers = self.centers_round
            on_off_pol = []
            type_all = []
            center_colors_all = []
            for n in range(self.n_neurons):
                is_on = W_sum[n, centers[n, 0], centers[n, 1]] > 0 #Fixed bug here, need to be cautious of x and y mapping
                center_colors = self.W[n, centers[n,0], centers[n,1],:]
                center_colors = self.d[:,n]*(1-self.c[:,n])
                center_colors_all.append(center_colors)
                L_center = center_colors[0]; M_center = center_colors[1]; S_center = center_colors[2]
                if L_center < 0 and M_center < 0 and S_center > 0:
                    color_center = 'blue'
                elif L_center > 0 and M_center > 0 and S_center < 0:
                    color_center = 'yellow'
                elif L_center < 0 and M_center < 0 and S_center < 0:
                    color_center = 'black'
                elif L_center > 0 and M_center > 0 and S_center > 0:
                    color_center = 'white'
                elif L_center > 0 and M_center < 0:
                    color_center = 'red'
                elif L_center < 0 and M_center > 0:
                    color_center = 'green'
                else:
                    color_center = 'unknown'
                type_all.append(color_center)
                    
                if is_on:
                    on_off_pol.append('ON')
                else:
                    on_off_pol.append('OFF')
            self.pathway = np.array(on_off_pol)
            self.type = np.array(type_all)
            self.center_colors = np.swapaxes(np.array(center_colors_all), 0, 1)