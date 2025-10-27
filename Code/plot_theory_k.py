#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 16:23:02 2025
@author: david
"""

from Fourier_theory import grid_models, power_fun, k0_global, power_predict2, Taylor_small1
import matplotlib.pyplot as plt

test = grid_models([100], [0.1])
a = test[0][0][0]
log_nu = a.log_nu
c = 2
A = 100#a.A[c]

plt.plot(Taylor_small1(a.freqs,k0_global,A,a.sigout,10**log_nu, sigin= a.sigin), label = r"Estimate when $\lambda$ << 1")
plt.plot(power_fun(a.freqs,k0_global,A,a.sigout,10**log_nu, sigin= a.sigin), label = "True function")
plt.xlabel("Frequency", size = 30)
plt.ylabel("Power", size = 30)
plt.xticks([])
plt.yticks([])
plt.legend(fontsize=30)
plt.title("A = " + str(A), size = 30)