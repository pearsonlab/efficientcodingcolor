# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 17:30:49 2024

@author: David
"""
fig, ax = plt.subplots(1)
line1, = ax.plot(new[0:199], label = "new")
line2, = ax.plot(old_a[0:199], label = "old a")
line3, = ax.plot(old_d, label = "old d")
line4, = ax.plot(more_var, label = "more var")
line5, = ax.plot(diff_init, label = "diff init")
line6, = ax.plot(original, label = "original")
ax.legend(handles=[line1,line2,line3,line4,line5,line6], fontsize = 50)
plt.xlabel("Steps x100", size = 30)
plt.ylabel("Loss", size = 30)


fig,ax = plt.subplots(1)
line1, = ax.plot(new, label = "new")
line2, = ax.plot(old_a, label = "old a")
ax.legend(handles = [line1,line2], fontsize = 50)


