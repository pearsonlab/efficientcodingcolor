# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:02:16 2024

@author: David
"""

plot = plt.imshow(numpy.flip(hey.Cx_eigvects[50,50,:,:], axis = 1), cmap = 'PiYG')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=30)
plot.axes.set_yticks([0,1,2], labels = ['L', 'M', 'S'], size = 30)
plot.axes.set_xticks([0,1,2], labels = ['1st', '2nd', '3rd'], size = 30)



plot_psd(numpy.log10(hey.Cx_eigvals[:,0:125,2]), title = "Log(1st eigenvalue)", v = 10)
plot_psd(numpy.log10(hey.Cx_eigvals[:,0:125,1]), title = "Log(2nd eigenvalue)", v = 10)
plot_psd(numpy.log10(hey.Cx_eigvals[:,0:125,0]), title = "Log(3rd eigenvalue)", v = 10)


                     
                     
plot_psd(numpy.log10(hey.Cx_eigvals[:,0:125,2]/hey.Cx_eigvals[:,0:125,1]), title = "Log ratio of 1st and 2nd eigenvalues")
plot_psd(numpy.log10(hey.Cx_eigvals[:,0:125,2]/hey.Cx_eigvals[:,0:125,0]), title = "Log ratio of 1st and 3rd eigenvalues", v = 5)
plot_psd(numpy.log10(hey.Cx_eigvals[:,0:125,1]/hey.Cx_eigvals[:,0:125,0]), title = "Log ratio of 2nd and 3rd eigenvalues", v = 5)

plot_psd(numpy.log10(abs(hey.Cx_bin[:,0:125,2,2])), title = "Log(S channel variance)", v = 10)



#X Y plot
plot = plt.imshow(numpy.log10(abs(hey.Cx[100,0,0,0:50,0:50])), origin = 'lower', cmap = 'YlOrRd')
plot.axes.set_xticks([])
plot.axes.set_yticks([])
plt.xlabel("X Spatial Frequency", size = 50)
plt.ylabel("Y Spatial Frequency", size = 50)