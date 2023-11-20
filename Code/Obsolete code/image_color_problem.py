# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:06:11 2023

@author: Rache
"""

img = scale(test.images.images[35].cpu().detach().numpy())
red = np.copy(img[0,:,:])
green = np.copy(img[1,:,:])
blue = np.copy(img[2,:,:])

img2 = np.zeros([img.shape[1],img.shape[2],3])
img2[:,:,0] = blue
img2[:,:,1] = green
img2[:,:,2] = red

#img2[:,:,0] = img[0,:,:]
#img2[:,:,1] = img[1,:,:]
#img2[:,:,2] = img[2,:,:]

plt.imshow(img2)