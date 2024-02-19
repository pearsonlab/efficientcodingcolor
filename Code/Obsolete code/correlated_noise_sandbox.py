# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 10:31:26 2024
@author: Rache
"""

import numpy as np
import math
import matplotlib.pyplot as plt

length = 10
x, y = np.meshgrid(np.array(range(length)), np.array(range(length)))

x1, y1 = x.flatten(), y.flatten()
x2, y2 = np.copy(x1), np.copy(y1)
B = 5
cov = np.zeros([length**2, length**2])
distances = np.zeros([length**2, length**2])
for pos1 in range(x1.shape[0]):
    for pos2 in range(x2.shape[0]):
        distance = np.sqrt((x1[pos1]-x2[pos2])**2 + (y1[pos1]-y2[pos2])**2)
        cov[pos1, pos2] = math.exp(-distance/B)
        distances[pos1, pos2] = distance



#B = np.ones([length**2,length**2])*0.01
#roll = np.repeat(np.array(range(length))[np.newaxis,:], repeats = length**2, axis = 0)



#def gauss_dist(x1,y1,roll, B):
#    x2 = np.roll(x1,roll)
#    y2 = np.roll(y1,roll)
#    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
#    return math.exp(-distance/B)


#result = list(map(gauss_dist,x1,y1,roll, B))







