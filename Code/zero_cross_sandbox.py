# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:53:30 2024

@author: David
"""
self = test1
step = 0.1
x = torch.arange(0, self.kernel_size, step)
y = torch.arange(0, self.kernel_size, step)
kernel_x = torch.tensor(9*np.ones(self.kernel_centers[:, 0].shape[0]))
kernel_y = torch.tensor(9*np.ones(self.kernel_centers[:, 1].shape[0]))

grid_x, grid_y = torch.meshgrid(x, y)
grid_x = grid_x.flatten().float()
grid_y = grid_y.flatten().float()
dx = kernel_x[None, :] - grid_x[:, None]
dy = kernel_y[None, :] - grid_y[:, None]
dist = (dx**2 + dy**2)
dog = shapes.DifferenceOfGaussianShape(kernel_size = test1.kernel_size, n_colors = test1.n_colors, num_shapes = test1.n_neurons, init_params = test1.all_params, read= True).shape_function(torch.tensor(dist))
dog = dog.reshape([test1.n_colors, test1.kernel_size,test1.kernel_size,test1.n_neurons])
dog = np.swapaxes(dog.detach().cpu().numpy(),0,3)




zero_cross = np.zeros(test1.n_neurons)
for n in range(test1.n_neurons):
    not_crossed = True
    first = np.max(abs(test1.rad_avg[n,:,0]))
    for r in range(0,18,101):
        dog = shapes.DifferenceOfGaussianShape(kernel_size = test1.kernel_size, n_colors = test1.n_colors, num_shapes = test1.n_neurons, init_params = test1.all_params, read= True).shape_function(torch.tensor([r]))
        if abs(first)*0.1 > abs(largest_now) and not_crossed:
            zero_cross[n] = r
            not_crossed = False
            
        #largest_prev = max(maxes[r-1], mins[r-1], key = abs)
        #if (largest_now >= 0) != (largest_prev >= 0):# or (-first*0.05 < largest_now < first*0.05):
        #    zero_cross[n] = r
        #    break
        
            