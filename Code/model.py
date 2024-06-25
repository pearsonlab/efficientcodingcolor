import os
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from shapes import get_shape_module, Shape
import matplotlib.pyplot as plt
from analysis_utils import closest_divisor, reshape_flat_W
from util import hexagonal_grid
import math

class Encoder(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 neurons: int,
                 nonlinearity: str,
                 input_noise: float,
                 output_noise: float,
                 shape: Optional[str],
                 individual_shapes: bool,
                 data_covariance,
                 fix_centers: bool,
                 n_colors: int,
                 n_mosaics: int,
                 corr_noise_sd: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.image_channels = n_colors
        self.D = kernel_size * kernel_size
        self.J = neurons
        self.nonlinearity = nonlinearity
        self.input_noise = input_noise
        self.output_noise = output_noise
        self.shape = shape
        self.register_buffer("data_covariance", data_covariance, persistent=False)
        self.fix_centers = fix_centers
        self.n_mosaics = n_mosaics
        
        if corr_noise_sd != 0 and corr_noise_sd is not None:
            self.nx_matrix(corr_noise_sd)
        if shape is not None:
           kernel_x = torch.rand(self.J) * (kernel_size - 1) / 2.0 + (kernel_size - 1) / 4.0
           kernel_y = torch.rand(self.J) * (kernel_size - 1) / 2.0 + (kernel_size - 1) / 4.0
           kernel_x[:2].fill_((kernel_size - 1) / 2.0)
           kernel_y[:2].fill_((kernel_size - 1) / 2.0)
           #kernel_x shape is [100]
           
           
    


           if not fix_centers:
               self.kernel_centers = nn.Parameter(torch.stack([kernel_x, kernel_y], dim=1))
               
           else:
               self.kernel_centers = hexagonal_grid(self.J, self.kernel_size, n_mosaics)

           assert self.J % 2 == 0, "only even numbers are allowed for 'neurons'"
           self.register_buffer("kernel_polarities", torch.tensor([-1, 1] * (self.J // 2)))
           shape_module = get_shape_module(shape)

           self.shape_function = shape_module(kernel_size, self.image_channels, self.J if individual_shapes else 1)
        
        else:  
            #David: added self.image_channels dimension so we have enough weights for cones
            W = 0.02 * torch.randn(self.image_channels*self.D, self.J)
            self.W = nn.Parameter(W / W.norm(dim=0, keepdim=True))  # spatial kernel, [D, J]
        
        self.logA = nn.Parameter(0.02 * torch.randn(self.J))  # gain of the nonlinearity
        self.logB = nn.Parameter(0.02 * torch.randn(self.J) - 1)  # bias of the nonlinearity
        #  self.nx_cov = self.correlated_input_noise()
        self.test_counter = 0
        
        #self.params = nn.ModuleDict({
        #    'MI': nn.ModuleList([self.W]),
        #    'firing': nn.ModuleList([self.logA, self.logB])})

    def kernel_variance(self):
    
        W = self.W
        W = self.W / self.W.norm(dim=0, keepdim=True)
        W = W.reshape(self.image_channels, self.kernel_size, self.kernel_size, self.J).mean(dim=0)
        Wx = W.pow(2).sum(dim=1)
        Wy = W.pow(2).sum(dim=0)

        coordsX = torch.arange(self.kernel_size, dtype=torch.float32, device=W.device)[:, None]
        meanWx = torch.sum(coordsX * Wx, dim=0)
        varWx = torch.sum((coordsX - meanWx).pow(2) * Wx, dim=0)
        coordsY = torch.arange(self.kernel_size, dtype=torch.float32, device=W.device)[:, None]
        meanWy = torch.sum(coordsY * Wy, dim=0)
        varWy = torch.sum((coordsY - meanWy).pow(2) * Wy, dim=0)
        
        return (varWx + varWy).mean()

    def jitter_kernels(self, power=1.0):
        with torch.no_grad():
            
            self.W.mul_(self.W.abs().pow(power))
            self.normalize()
            

    def spatiotemporal(self, input: torch.Tensor):
        y = input @ self.W
        return y

    def matrix_spatiotemporal(self, input: torch.Tensor, gain: torch.Tensor, cov = False, record_C = False):
        # compute C_rx in VAE note page 8.
        # input.shape = [LD, LD], gain.shape = [1 or B, T or 1, J]
        assert input.ndim == 2 and input.shape[0] == input.shape[1]
        L = input.shape[0] // self.D
        D = self.D
        J = self.J
        #L = 3, D = 324, J = 100
        
        #David: I permuted x multiple once so the tensor multiplication 
        #(y = input @ self.w from spatiotemporal) would have consistent dimensions. 
        #print(f"shape input = {input.shape}")
        x = self.spatiotemporal(input)             # shape = [D, 1, J] or [LDprint(), T, J]
        x = x.permute(1, 0)                 # shape = [1, J, D] or [T, J, LD]
        output_dim = x.shape[0]
        x = self.spatiotemporal(x)             # shape = [J, 1, J] or [TJ, T, J]
        if record_C:
            self.WCxW = x
        #David: gain.shape = [3,100,100]
        G = gain.reshape(-1, output_dim)       # shape = [1 or B, J] or [1 or B, TJ]
 
        x = G[:, :, None] * x * G[:, None, :]  # shape = [1 or B, J, J] or [1 or B, TJ, TJ]
        
        return x  # this is C_rx
    
        def make_kernel_centers(self):
            n_neurons = self.J
            n_dist, n_angles = closest_divisor(n_neurons)
            max_length = int(self.kernel_size/2)
            all_dist = np.linspace(0,max_length, n_dist)
            all_angles = np.linspace(0, np.pi*2, n_angles)
            all_kernel_centers = np.zeros([n_dist, n_angles, 2])
            for dist_index in range(n_dist):
                for angle_index in range(n_angles):
                    angle = all_angles[angle_index]
                    dist = all_dist[dist_index]
                    angle_dist = abs(all_angles[1] - all_angles[0])
                    if dist_index%2 == 0:
                        phase = 0
                    else:
                        phase = angle_dist
                    x = np.cos(angle + phase)*dist
                    y = np.sin(angle + phase)*dist
                    all_kernel_centers[dist_index,angle_index, 0] = x
                    all_kernel_centers[dist_index,angle_index, 1] = y
            return all_kernel_centers


    def normalize(self):
        with torch.no_grad():
            self.W /= self.W.norm(dim=0, keepdim=True)
            
    
    def nx_matrix(self, B):
        x, y = np.meshgrid(np.array(range(self.kernel_size)), np.array(range(self.kernel_size)))

        x1, y1 = x.flatten(), y.flatten()
        x2, y2 = np.copy(x1), np.copy(y1)
        cov = np.zeros([self.kernel_size**2, self.kernel_size**2])
        for pos1 in range(x1.shape[0]):
            for pos2 in range(x2.shape[0]):
                distance = np.sqrt((x1[pos1]-x2[pos2])**2 + (y1[pos1]-y2[pos2])**2)
                cov[pos1, pos2] = math.exp(-distance/B)
        
        self.nx_cov = cov
        self.nx_L = torch.tensor(np.linalg.cholesky(cov), dtype = torch.float32, device = "cuda:0")

    def forward(self, image: torch.Tensor, firing_restriction, corr_noise_sd, record_C = False):
        D = self.D
        L = image.shape[1]
        B = image.shape[0]
        
        if self.shape is not None:
            self.W = self.shape_function(self.kernel_centers, self.kernel_polarities)
        gain = self.logA.exp()  # shape = [J]
        bias = self.logB.exp()
        
        if corr_noise_sd == 0 or corr_noise_sd == None:
            nx = self.input_noise * torch.randn_like(image) #David: potential bug here
        else:
            nx = torch.matmul(self.input_noise * torch.randn([B, L, self.D], device = "cuda:0"), torch.transpose(self.nx_L,0,1))

        #David: I permuted image so the tensor multiplication 
        #(y = input @ self.w from spatiotemporal) would have consistent dimensions. 
        image_nx = image + nx
        self.test_counter = self.test_counter + 1
        image_nx = image_nx.flatten(1,2)
        #image_nx = image_nx.reshape([25,L*D]) #This is buggy, flattening is much better

        y = self.spatiotemporal(image_nx)
        nr = self.output_noise * torch.randn_like(y)
        z = gain * (y - bias) + nr  # z.shape = [B, T, J] #There is a gain here

        
        if self.nonlinearity == "relu":
            r = gain * (y - bias).relu() #Removed gain here
            
            grad = ((y - bias) > 0).float()  # shape = [B, T, J]
        else:  # softplus nonlinearity
            r = gain * F.softplus(y - bias, beta=2.5)
            grad = torch.sigmoid(2.5 * (y-bias))
        gain = gain*grad
        C_nx = self.input_noise ** 2 * torch.eye(L * D, device=image.device)
        C_zx = self.matrix_spatiotemporal(C_nx, gain, record_C = False)  # shape = [1 or B, J, J] or [1 or B, TJ, TJ]
        assert C_zx.shape[1] == C_zx.shape[2]
        C_nr = self.output_noise ** 2 * torch.eye(C_zx.shape[-1], device=image.device)
        C_zx += C_nr
        C_z = self.matrix_spatiotemporal(self.data_covariance + C_nx, gain, cov = True, record_C = True)
        C_z += C_nr
        
    
        self.C_z, self.C_zx = torch.mean(C_z,dim= 0), torch.mean(C_zx,dim = 0)
        
        
        return z, r, C_z, C_zx



@dataclass
class OutputMetrics(object):
    KL: torch.Tensor = None
    loss: torch.Tensor = None
    linear_penalty: torch.Tensor = None
    quadratic_penalty: torch.Tensor = None
    h: torch.Tensor = None

    def final_loss(self, firing_restriction):
        if firing_restriction == "Lagrange":
            return self.loss.mean() + self.linear_penalty + self.quadratic_penalty
        else:
            return self.loss.mean()
    
    def return_h(self):
        return self.h


class OutputTerms(object):
    logdet_numerator: torch.Tensor = None
    logdet_denominator: torch.Tensor = None
    r_minus_one_squared = None

    z: torch.Tensor = None
    r: torch.Tensor = None

    def __init__(self, model: "RetinaVAE"):
        self.model = model

    def calculate_metrics(self, i, firing_restriction) -> "OutputMetrics":
        KL = self.logdet_numerator - self.logdet_denominator

        target = os.environ.get("FIRING_RATE_TARGET", "1")
        if 'i' in target:
            target = eval(target)
        else:
            target = float(target)
        if self.model.Lambda.shape[0] == 1:
            #h = self.r.sub(target).mean()  # Original version by Nayoung
            h = (self.r-1).mean()
        else:
            #h = self.r.sub(target).mean(dim=0)  # Original version by Nayoung
            h = (self.r-1).mean(dim=0)
        

        if firing_restriction == "Lagrange":
            linear_penalty = (self.model.Lambda * h).sum()
            quadratic_penalty = self.model.rho / 2 * (h ** 2).sum()
        else:
            linear_penalty = 0
            quadratic_penalty = 0

        return OutputMetrics(
            KL=KL,
            loss=self.model.beta * KL,
            linear_penalty=linear_penalty,
            quadratic_penalty=quadratic_penalty,
            h = h
        )


class RetinaVAE(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 neurons: int,
                 input_noise: float,
                 output_noise: float,
                 nonlinearity: str,
                 shape: Optional[str],
                 individual_shapes: bool,
                 beta: float,
                 rho: float,
                 data_covariance,
                 fix_centers,
                 n_colors,
                 n_mosaics,
                 corr_noise_sd):  
        super().__init__()
        self.beta = beta
        self.rho = rho
        self.D = kernel_size * kernel_size

        assert nonlinearity in {"relu", "softplus"}
        self.encoder = Encoder(kernel_size, neurons, nonlinearity, input_noise, output_noise, shape, individual_shapes, data_covariance, fix_centers, n_colors, n_mosaics, corr_noise_sd)

        self.Lambda = nn.Parameter(torch.rand(neurons))

    def forward(self, x, firing_restriction, corr_noise_sd = 0, record_C = False) -> OutputTerms:
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.D)  # x.shape = [B, L, D] (L: input time points)
        
        o = OutputTerms(self)
        o.z, o.r, numerator, denominator = self.encoder(x, firing_restriction, corr_noise_sd, record_C = record_C)

        if numerator is not None:
            L_numerator = numerator.cholesky()
            o.logdet_numerator = 2 * L_numerator.diagonal(dim1=-1, dim2=-2).log2().sum(dim=-1)

        if denominator is not None:
            L_denominator = denominator.cholesky()
            o.logdet_denominator = 2 * L_denominator.diagonal(dim1=-1, dim2=-2).log2().sum(dim=-1)

        return o
