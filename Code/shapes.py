
import torch
from torch import nn

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#matplotlib.use('QtAgg')

class Shape(nn.Module):
    def __init__(self, kernel_size, initial_parameters, num_shapes, n_colors, read = False):
        super().__init__()
        x = torch.arange(kernel_size)
        y = torch.arange(kernel_size)
        grid_x, grid_y = torch.meshgrid(x, y)
        self.kernel_size = kernel_size
        self.register_buffer("grid_x", grid_x.flatten().float())
        self.register_buffer("grid_y", grid_y.flatten().float())
        self.n_colors = n_colors
        if n_colors >= 1 and not read:
            params_pre = np.tile(initial_parameters, n_colors)
            params = torch.tensor(params_pre).unsqueeze(-1).repeat(1, num_shapes)
            for p in range(params.shape[0]):
                print(params.shape, "params.shape")
                params[p, :] = torch.normal(mean=params[p,:], std = torch.tensor(1.0).repeat(num_shapes))      
        else:
            params = torch.tensor(initial_parameters)
        #print(torch.var(params, dim = 1))
        #params = torch.tensor(params_pre).unsqueeze(-1).repeat(1, num_shapes)

        self.shape_params = nn.Parameter(params, requires_grad=True)
        
    def forward(self, kernel_centers, kernel_polarities = None, normalize=True):
        kernel_x = kernel_centers[:, 0]
        kernel_y = kernel_centers[:, 1]
        #print(kernel_x.device, self.grid_x.device, "Here I am!")
        dx = kernel_x[None, :] - self.grid_x[:, None]
        dy = kernel_y[None, :] - self.grid_y[:, None]

        W = self.shape_function(dx ** 2 + dy ** 2)
        if normalize:
            W = W / W.norm(dim=0, keepdim=True)

        return W #* kernel_polarities

    def shape_function(self, rr):
        raise NotImplementedError


class DifferenceOfGaussianShape(Shape):
    def __init__(self, kernel_size, n_colors, num_shapes=1, init_params = [-0.5, -1, -0.5, 0.0], read = False):
        super().__init__(kernel_size, init_params, num_shapes, n_colors, read)
    

    def shape_function(self, rr):   
        n_params = self.shape_params.shape[0]
        logA_index = np.array(range(0,n_params,4))
        logB_index = logA_index + 1
        logitC_index = logA_index + 2
        D_index = logA_index + 3

        logA = self.shape_params[logA_index]; logB = self.shape_params[logB_index]; 
        logitC = self.shape_params[logitC_index]; d = self.shape_params[D_index]

        a_pre = logA.exp()
        b = logB.exp()
        a = a_pre + b  # make the center smaller than the surround
        c = logitC.sigmoid()  #to keep it within (0, 1)
        d = d/torch.sqrt(torch.sum(d**2, 0))
        self.a, self.b, self.c, self.d = a.detach(), b.detach(), c.detach(), d.detach()
        
        a = torch.unsqueeze(a,0)
        b = torch.unsqueeze(b,0)
        rr = torch.unsqueeze(rr,1)
        rr = rr.repeat(1,self.n_colors,1)
        #print('a:',a[0],'b',b[0], 'logA', logA[0], 'logB', logB[0])
        DoG = d*(torch.exp(-a * rr) - c * torch.exp(-b * rr))
        DoG = torch.swapaxes(DoG, 0, 1) #David: Without this line, output of this module and the shape of the stimuli don't match. Important bug that took multiple weeks to fix. 
        self.W = DoG
        DoG = DoG.flatten(0,1)
        return DoG.float()


class GaussianShape(Shape):
    def __init__(self, kernel_size, num_shapes=1):
        super().__init__(kernel_size, [-0.75], num_shapes)

    def shape_function(self, rr):
        self.a = self.shape_params.exp()
        return torch.exp(-self.a * rr)


class DifferenceOfTDistributionShape(Shape):
    def __init__(self, kernel_size, num_shapes=1):
        super().__init__(kernel_size, [-3, -0.9, 0], num_shapes)

    def shape_function(self, rr):
        logA, logB, logitlogC = self.shape_params
        a = logA.exp()
        b = logB.exp()
        a = a + b  # make the center smaller than the surround
        max_r = self.kernel_size // 4
        logitlogC = self.shape_params[2]
        logC = - (a - b) * max_r ** 2 * logitlogC.sigmoid()  #to keep it within (0, 1)
        c = logC.exp()
        self.a, self.b, self.c = a.detach(), b.detach(), c.detach()
        nu = 1
        return (1 + a * rr / nu) ** (-(nu + 1) / 2) - c * (1 + b * rr / nu) ** (-(nu + 1) / 2)


class SingleTDistribution(Shape):
    def __init__(self, kernel_size, num_shapes=1):
        super().__init__(kernel_size, [-3], num_shapes)

    def shape_function(self, rr):
        logA = self.shape_params
        a = logA.exp()
        self.a = a.detach()
        nu = 2
        return (1 + a * rr / nu) ** (-(nu + 1) / 2)


def get_shape_module(type):
    return {
        'difference-of-gaussian': DifferenceOfGaussianShape,
        'gaussian': GaussianShape,
        'difference-of-t': DifferenceOfTDistributionShape,
        'single-t': SingleTDistribution,
    }[type]

