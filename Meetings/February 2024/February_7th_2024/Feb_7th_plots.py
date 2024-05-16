# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 22:09:15 2024

@author: David
"""

max_d = np.argmax(abs(test.d), axis = 0)

c_og = test.c
c_fit = test.DoG_mod.c.detach().cpu().numpy()

c_og_max = np.choose(max_d, c_og)
c_fit_max =  np.choose(max_d, c_fit)

a_og = np.choose(max_d, test.a)
a_fit = np.choose(max_d, test.DoG_mod.a.detach().cpu().numpy())

b_og = np.choose(max_d, test.b)
b_fit = np.choose(max_d, test.DoG_mod.b.detach().cpu().numpy())

plt.plot(c_fit_max, c_og_max, 'o')
plt.plot(np.arange(0,1,0.01), np.arange(0,1,0.01))
plt.xlabel("Fitted c parameter", size = 50)
plt.ylabel("c parameter", size = 50)


plt.plot(a_fit,a_og, 'o')
plt.ylim([0,2]); plt.xlim([0,2])
plt.plot(np.arange(0,2,0.01), np.arange(0,2,0.01))
plt.xlabel("Fitted a parameter", size = 50)
plt.ylabel("a parameter", size = 50)

plt.plot(b_fit,b_og, 'o')
plt.plot(np.arange(0,1,0.01), np.arange(0,1,0.01))
plt.xlabel("Fitted b parameter", size = 50)
plt.ylabel("b parameter", size = 50)


#######################################################

fig, axes = plt.subplots(1,3)
line = np.arange(0,2,0.01)
        
r_or_g = np.logical_or(max_d == 1, max_d == 0)
s = max_d == 2

#r_or_g = np.repeat(True, 300)
#s = np.repeat(True, 300)

axes[0].plot(test.a[0,r_or_g], test.b[0,r_or_g], 'o')
axes[0].set_ylim([0,2])
axes[0].set_xlim([0,2])
axes[0].plot(line,line)
axes[0].set_title("L inputs", size = 30)

axes[1].plot(test.a[1,r_or_g], test.b[1,r_or_g], 'o')
axes[1].set_ylim([0,2])
axes[1].set_xlim([0,2])
axes[1].plot(line,line)
axes[1].set_title("M inputs", size = 30)

axes[2].plot(test.a[2,max_d == 2], test.b[2,max_d == 2], 'o')
axes[2].set_ylim([0,2])
axes[2].set_xlim([0,2])
axes[2].plot(line,line)
axes[2].set_title("S inputs", size = 30)

fig.supxlabel('Precision of the center', size = 50)
fig.supylabel('Precision of the surround', size = 50)

########################
fig, axes = plt.subplots(1,3)
d_type = np.stack([r_or_g, r_or_g, s], axis = 1)

for n in range(3):
    axes[n].hist(test.b[n,d_type[:,n]]/test.a[n,d_type[:,n]], bins = np.arange(0,5,0.01))
    axes[n].set_xlim([0,2])
    axes[n].set_ylim([0,50])
    



