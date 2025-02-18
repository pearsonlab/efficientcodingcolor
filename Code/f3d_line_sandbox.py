#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:47:54 2025

@author: david"""

plt.close()
#frames = [10450,10500] #has a line 
frames = [5500,5550] #does not have a line
frames = [1700,1750] #has a line
#frames = [28000,28050] #has multiple lines
#frames = [15000, 15050] # has a thick line
frames = [15500,15550]

segment = video_segment(hey.video_clip, frames, hey.color_means, hey.color_stds)
segment.make_psd_3D()

f3D = segment.f3D.get()
f3D_v2 = numpy.copy(f3D)
#f3D_v2[:,:,0,:] *= 2
f3D_v2[:,:,:,0] *= 5

psd_v1 = f3D*numpy.conjugate(f3D)



img_v1 = numpy.real(numpy.fft.fftn(f3D, axes = (0,2,3)))
img_v2 = numpy.real(numpy.fft.fftn(f3D_v2, axes = (0,2,3)))

img_v1 = numpy.flip(img_v1, axis = 2)
img_v2 = numpy.flip(img_v2, axis = 2)

plt.imshow(img_v1[int(len(frames)/2),0,:,:])
plt.figure()
plt.imshow(img_v2[int(len(frames)/2),0,:,:])
plt.figure()
plt.imshow(numpy.log10(segment.psd3D[0,0,:,:].get()).T)
plt.figure()
plt.imshow((numpy.log10(abs(f3D_v2)**2)[0,0,0:50,0:50].T))
#plt.imshow(img_v2[50,0,:,:])



#Cx = test['Cx'][0,0,0,0:128,0:128]
#line = []
#for i in range(Cx.shape[0]):
#    for j in range(Cx.shape[1]):
#        line.append(Cx[i,i])
