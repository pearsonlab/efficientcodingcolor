# -*- coding: utf-8 -*-
"""
Created on Wed May 22 21:25:48 2024

@author: David
"""
import numpy as np
import matplotlib.pyplot as plt

plt.gca().set_aspect('equal')
plt.plot(np.arange(-4,5), np.concatenate((np.flip(np.arange(1,5)), np.arange(0,5))), color = 'black', linewidth = 5)
plt.axvline(0, linestyle = '--')
plt.xlabel("Input", size = 100)
plt.ylabel("Output", size = 100)
plt.xticks(fontsize=75)
plt.yticks(fontsize=75)
plt.yticks(ticks=[0,1,2,3,4], labels=np.array([0,1,2,3,4]).astype(int))