# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:16:54 2024

@author: David
"""
params = test.model.encoder.shape_function.shape_params.detach()
for n in range(test.n_neurons):
    if test.type[n] == 1:
        params[3,n] *= 2
        params[7,n] = 0
        #rad[n,:,0] *= -1
#    if test.type[n] == 3:
#        params[7,n] *= -1
    
params.requires_grad = True
test.model.encoder.shape_function.shape_params = nn.Parameter(params)
test.cp['model_state_dict']['encoder.shape_function.shape_params'] = nn.Parameter(params)


torch.save(test.model, path + "test_model.pt")
torch.save(test.cp, path + "test_cp.pt")