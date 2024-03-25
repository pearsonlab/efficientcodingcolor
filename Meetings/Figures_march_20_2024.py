# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 22:49:05 2024

@author: David
"""

fig, ax = plt.subplots(1)
line1, = ax.plot(test_all.MI, label = "MI")

line2, = ax.plot(test_all.losses, label = "Loss")

ax.legend(handles = [line1,line2], fontsize = 50)
plt.xlabel("Steps", size = 50)


##### Show that two mosaics from different runs complement each other
fig, ax = plt.subplots(1)
ax.set_aspect('equal')
ax.set_facecolor('tab:gray')
ax.set_xlim([0,18])
ax.set_ylim([0,18])
ax.set_yticklabels([])
ax.set_xticklabels([])
kernel_centers1 = test.kernel_centers[test.type == 3,:]
kernel_centers2 = test2.kernel_centers[test2.type == 4,:]
kernel_centers = np.concatenate((kernel_centers1, kernel_centers2))
for n in range(kernel_centers.shape[0]):
    marker = 'o'
    x = kernel_centers[n, 0]
    y = kernel_centers[n, 1]
    ax.plot(x, y, marker = marker, markersize = 12, color = 'green')
