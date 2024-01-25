"""
Created on Mon Jan 22 17:10:50 2024
@author: Rache
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Qtagg")

x1 = torch.tensor(1.5, requires_grad = True)
x2 = torch.tensor(0.0, requires_grad = True)

z_all = []
x1_all = []

x2_fixed = 1
#Define covariance matrix thats oriented
cov = nn.Parameter(torch.tensor([[1,0.7], [0.7,1]]), requires_grad = False)

#X and Y mean of the multivariate gaussian
u = torch.tensor([0.0,0.0], requires_grad = False)

#Return z values from x1 and x2
def forward(x):
    return torch.exp(-torch.matmul(torch.matmul(x-u,cov),x-u))

#z.requires_grad = True
optimizer = torch.optim.SGD([x1,x2], lr = 0.1)
optimizer2 = torch.optim.SGD([x2], lr = 0.2)
for i in range(10000):
    x = torch.stack((x1,x2))
    optimizer.zero_grad()
    optimizer2.zero_grad()
    #Let's start first optimization
    z = -forward(x)
    z_all.append(z.detach().cpu().numpy())
    x1_all.append(x1.item())
    z.backward(retain_graph = True)
    optimizer.step()
    optimizer.zero_grad()
    
    #Lets start second optimization
    loss_2 = (x2 - torch.tensor(x2_fixed))**2
    loss_2.backward()
    optimizer2.step()
    
    optimizer2.zero_grad()
    print(x[0].item(), x[1].item(), z)

plt.close()
plt.plot(x1_all, scaley = False)
plt.xlabel("Epochs", size = 10)
plt.ylabel("x1", size = 10)
plt.ylim(-5,5)

gaussian_values = np.zeros([100,100])
i_x = 0
for i in np.linspace(2,-2,100):
    j_x = 0
    for j in np.linspace(-2,2,100):
        gaussian_values[i_x,j_x] = forward(torch.tensor([i,j]).float())
        j_x = j_x + 1
    i_x = i_x+ 1
    
#plt.imshow(gaussian_values)
#plt.axvline(70, linewidth = 3)



