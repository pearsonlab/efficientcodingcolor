import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("agg")
np.set_printoptions(suppress=True)

def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def kernel_images(W, kernel_size, image_channels=1, rows=None, cols=None, spacing=1):
    """
    Return the kernels as tiled images for visualization
    :return: np.ndarray, shape = [rows * (kernel_size + spacing) - spacing, cols * (kernel_size + spacing) - spacing, 1]
    """

    W /= np.linalg.norm(W, axis=0, keepdims=True)
    W = W.reshape(image_channels, -1, W.shape[-1])

    if rows is None:
        rows = int(np.ceil(math.sqrt(W.shape[-1])))
    if cols is None:
        cols = int(np.ceil(W.shape[-1] / rows))

    kernels = np.ones([image_channels, rows * (kernel_size + spacing) - spacing, cols * (kernel_size + spacing) - spacing], dtype=np.float32)
    coords = [(i, j) for i in range(rows) for j in range(cols)]

    Wt = W.transpose(2, 0, 1)

    for (i, j), weight in zip(coords, Wt):
        kernel = weight.reshape(image_channels, kernel_size, kernel_size) * 2 + 0.5
        x = i * (kernel_size + spacing)
        y = j * (kernel_size + spacing)
        kernels[:, x:x+kernel_size, y:y+kernel_size] = kernel

    return kernels.clip(0, 1)


def get_matrix(matrix_type):
    pca_comps = np.array([[ 0.51876956, 0.52288215, 0.67636706],
                 [ 0.48552343, 0.4709887, -0.73650299],
                 [ 0.7036655, -0.71046739, 0.00953693]])
    pca_inv = np.linalg.inv(pca_comps)  
    
    rgb_to_lms = np.array([[17.88240413, 43.51609057,  4.11934969],
             [ 3.45564232, 27.15538246,  3.86713084],
             [ 0.02995656,  0.18430896,  1.46708614]])
    lms_to_rgb = np.linalg.inv(rgb_to_lms)
    
    if matrix_type == 'pca_comps':
        return pca_comps
    elif matrix_type == 'pca_inv':
        return pca_inv
    elif matrix_type == 'rgb_to_lms':
        return rgb_to_lms
    elif matrix_type == 'lms_to_rgb':
        return lms_to_rgb

def scale_0to1(W):
    W = (W - np.min(W))/(np.max(W) - np.min(W))
    return W

def hexagonal_grid(n_neurons, kernel_size, n_mosaics):
    if n_neurons%n_mosaics != 0:
        raise ValueError("Number of neurons has to be a multiple of the number of mosaics!")
    neurons_per_mosaic = int(n_neurons/n_mosaics)
    radius = kernel_size/2
    dist = 1
    goal_neurons = False
    while not goal_neurons:
        size = int(kernel_size/2)
        #n_x, n_y = closest_divisor(n_neurons)
        x_all = np.arange(-size,size,dist)
        y_all = np.arange(-size,size,dist)
        
        x, y = np.meshgrid(x_all, y_all)
        for x_pos in range(x.shape[1]):
            if x_pos%2 == 0:
                x[x_pos,:] = x[x_pos,:] - dist/2
        
        x_flat = x.flatten()
        y_flat = y.flatten()
        kernel_centers = np.stack((x_flat,y_flat),1)
    
        center_dist = np.sqrt(x_flat**2 + y_flat**2)
        within_radius = np.sum(center_dist < radius)
        if within_radius < neurons_per_mosaic:
            dist = dist - 0.001
        elif within_radius > neurons_per_mosaic:
            dist = dist + 0.001
        else:
            goal_neurons = True
        
    kernel_centers = kernel_centers[center_dist < radius, :] #Subset to neurons within radius
    kernel_centers = kernel_centers + radius #Change axes to follow [0,kernel_size] convention
    kernel_centers = torch.tensor(kernel_centers, device = 'cuda') #Tensor
    kernel_centers = kernel_centers.tile(n_mosaics,1) #Tile for every mosaic
    return kernel_centers