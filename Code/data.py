import os
from typing import Union

import numpy as np
import torch
from PIL import Image
from scipy.io.matlab import loadmat
from torch.utils.data import Dataset
from tqdm import trange, tqdm
from util import get_matrix
from sklearn.decomposition import PCA
import random as rnd



def circular_mask(kernel_size: int):
    """
    :param kernel_size:
    :return: masked image with slope with width 1?
    """
    radius = kernel_size / 2 - 0.5
    x = torch.linspace(-radius, radius, kernel_size)
    y = torch.linspace(-radius, radius, kernel_size)
    xx, yy = torch.meshgrid(x, y)
    mask = torch.clamp(radius - torch.sqrt(xx ** 2 + yy ** 2) + 1, min=0.0001, max=1)
    return mask


def estimated_covariance(dataset: Dataset, num_samples: int, device: Union[str, torch.device] = None, index=0):
    loop = trange(num_samples, desc="Taking samples for covariance calculation", ncols=99)

    samples = torch.stack([dataset[index].flatten() for _ in loop])  # / dataset.mask
    if device is not None:
        samples = samples.to(device)
    samples -= samples.mean(dim=0, keepdim=True)
    C = samples.t() @ samples / num_samples
    C = (C + C.t()) / 2.0  # make it numerically symmetric
    return C


class KyotoNaturalImages(Dataset):
    """
    A Torch Dataset class for reading the Kyoto Natural Image Dataset, available at:
        https://github.com/eizaburo-doi/kyoto_natim
    This dataset consists of 62 natural images in the MATLAB format, and each image has
    either 500x640 or 640x500 size, from which a rectangular patch is randomly extracted.
    """

    def __init__(self, root, kernel_size, circle_masking, device, n_colors, normalize_color, restriction = 'True', remove_mean = False):
        #root = 'kyoto_natim'
        #device = 'cuda'
        print('root', root)
        files = [mat for mat in os.listdir(root) if mat.endswith('.mat')]
        print("Loading {} images from {} ...".format(len(files), root))
        self.n_colors = n_colors
        self.normalize_color = normalize_color
        self.circle_masking = circle_masking
        self.remove_mean = remove_mean
        images = []
        L_means = 0
        L_stds = 0
        M_means = 0
        M_stds = 0
        S_means = 0
        S_stds = 0
        
        for file in tqdm(files):
            if file.endswith('.mat'):
                #David: Combined the responses from the three different cones into a single image array
                imageOM = loadmat(os.path.join(root, file))['OM'].astype(np.float64)
                imageOS = loadmat(os.path.join(root, file))['OS'].astype(np.float64)
                imageOL = loadmat(os.path.join(root, file))['OL'].astype(np.float64)
                
                #Inhibition from an "horizontal" cell to reduce MI between M and S cones. 
                #MS_tot = imageOM + imageOS
                #imageOM = imageOM - 0.38*MS_tot
                #imageOS = imageOS - 0.38*MS_tot
                
                pca_comps = get_matrix('pca_comps')
                fake_channel = np.random.normal(loc = 0, scale = 0.01, size=imageOL.shape)
                
                if n_colors == 1:
                    image = imageOM
                if n_colors == 2:
                    image = np.array([imageOL, imageOS])
                elif n_colors == 3:
                    image = np.array([imageOL, imageOM, imageOS])
                else:
                    Exception("You can only have 1 or 3 colors")
                #image = np.tensordot(pca_comps, image, axes = 1)
                
            else:
                image = np.array(Image.open(os.path.join(root, file)).convert('L')).astype(np.float64)
                
            std = np.std(image)
            if std < 1e-4: #This line never gets called 
                continue
            
            L_mean = np.mean(image[0,:,:])
            L_std = np.std(image[0,:,:])
            M_mean = np.mean(image[1,:,:])
            M_std = np.std(image[1,:,:])
            
            if image.shape[0] > 2:
                S_mean = np.mean(image[2,:,:])
                S_std = np.std(image[2,:,:])
                S_means += S_mean
                S_stds += S_std
                if normalize_color:
                    image[2,:,:] -= S_mean
                    image[2,:,:] /= S_std
                    
            if normalize_color:
                image[0,:,:] -= L_mean
                image[0,:,:] /= L_std
                image[1,:,:] -= M_mean
                image[1,:,:] /= M_std
            else:
                image -= np.mean(image)
                image /= std
                
                 
            L_means += L_mean
            L_stds += L_std
            M_means += M_mean
            M_stds += M_std
            
                

            
            
            #print('start', np.mean(image), np.std(image), image.shape, np.mean(image[0,:,:]), np.mean(image[1,:,:]), np.std(image[0,:,:]), np.std(image[1,:,:]))
            
            #Uncomment these lines when you're done plz

            
            
            
            images.append(torch.from_numpy(image).to(device))
        L_means /= len(images)
        M_means /= len(images)
        S_means /= len(images)
        L_stds /= len(images)
        M_stds /= len(images)
        S_stds /= len(images)
        
        #for i in range(len(images)):
         #   images[i][0,:,:] -= L_means
          #  images[i][1,:,:] -= M_means
           # images[i][2,:,:] -= S_means
           # images[i][0,:,:] /= L_stds
           # images[i][1,:,:] /= M_stds
           # images[i][2,:,:] /= S_stds
            
        self.device = device
        self.images = images
        self.kernel_size = kernel_size
        self.restriction = restriction
        self.remove_mean = remove_mean
        if isinstance(kernel_size, int):
            self.mask = circular_mask(kernel_size) if circle_masking else torch.ones((kernel_size, kernel_size))
        else:
            self.mask = torch.ones([kernel_size[0], kernel_size[1]])
        self.mask = self.mask.to(device)

    def __len__(self):
        """Returns 100 times larger than the number of images, to enable batches larger than 62"""
        return len(self.images) * 100

    def __getitem__(self, index):
        """Slices an [dx, dy] image at a random location"""
        while True:
            index = np.random.randint(len(self.images))
            image = self.images[index]
            dx, dy = self.kernel_size, self.kernel_size
            x = np.random.randint(image.shape[-2] - dx)
            y = np.random.randint(image.shape[-1] - dy)
            result = image[..., x:x+dx, y:y+dy] * self.mask
            condition = eval(self.restriction)       
            if condition:
                if self.remove_mean:
                    for i in range(result.shape[0]):
                        result[i,:,:] = result[i,:,:] - torch.mean(result[i,:,:])
                return result.float()

    def covariance(self, num_samples: int = 100000, device: Union[str, torch.device] = None, index=0):
        return estimated_covariance(self, num_samples, device, index)
    
    def pca_color(self):
        self.n_images = len(self.images)
        images_reshaped = np.zeros([self.n_colors, 0], dtype = float)
        for n_color in range(self.n_colors):
            color = np.array([])
            for i in range(self.n_images):
                color = np.append(color, self.images[i][n_color,:,:].flatten().cpu())
            if n_color == 0:
                stack_length = color.shape[0]
                images_reshaped = np.zeros([self.n_colors, stack_length])
            images_reshaped[n_color,:] = color
        self.flat = np.transpose(images_reshaped)
        pca = PCA(n_components = self.n_colors)
        pca.fit(self.flat)
        self.pca = pca
        self.comps = pca.components_
        self.comps_inv = np.linalg.inv(self.comps)
        
        pcs = self.pca.fit_transform(self.flat)
        self.mean_pcs = np.mean(np.sqrt(pcs**2), axis = 0)
        
    def whiten_pca(self, ratio):
        image_index = 0
        for image in self.images:
            for color in range(self.n_colors):
                image_flat = torch.reshape(image, [self.n_colors, image.shape[1]*image.shape[2]])
                pcs = self.pca.fit_transform(np.transpose(image_flat.detach().cpu().numpy()))
                var = self.pca.explained_variance_
                A = []
                for i in range(self.n_colors):
                    A.append(np.sqrt(ratio[i]/var[i]))
                A = np.diag(A)
                #A = np.diag([1,1,1])
                image_whiten = np.matmul(np.matmul(np.transpose(self.pca.components_), A), np.transpose(pcs))
            
            image = np.reshape(image_whiten, image.shape)
            #std = np.std(image)
            #image -= np.mean(image)
            #image /= std
            self.images[image_index] = torch.tensor(image, device = self.device)
            image_index = image_index + 1
            
            
    
