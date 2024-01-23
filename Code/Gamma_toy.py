"""
Created on Mon Jan 22 17:10:50 2024
@author: Rache
"""

import torch
from torch import nn
import numpy as np

x1 = torch.tensor(2.0, requires_grad = True)
x2 = torch.tensor(1.0, requires_grad = True)

#y = torch.tensor(1.0, requires_grad = True)
cov = nn.Parameter(torch.tensor([[1,0.7], [0.7,1]]), requires_grad = False)
#xy = torch.tensor([x,y], requires_grad = True)
u = torch.tensor([1.0,1.5], requires_grad = False)

def forward(x):
    return torch.exp(-torch.matmul(torch.matmul(x-u,cov),x-u))

#z.requires_grad = True
optimizer = torch.optim.SGD([x1,x2], lr = 0.01)
optimizer2 = torch.optim.SGD([x2], lr = 0.2)
for i in range(1000):
    x = torch.stack((x1,x2))
    optimizer.zero_grad()
    optimizer2.zero_grad()
    #Let's start first optimization
    z = -forward(x)
    z.backward(retain_graph = True)
    optimizer.step()
    optimizer.zero_grad()
    
    #Lets start second optimization
    loss_2 = (x2 - torch.tensor(3))**2
    loss_2.backward()
    optimizer2.step()
    
    optimizer2.zero_grad()
    print(x[0].item(), x[1].item(), z)
    