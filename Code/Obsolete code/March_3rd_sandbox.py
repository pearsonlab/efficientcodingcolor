# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 17:37:23 2024

@author: David
"""
input_noise = 0.4
output_noise = 1
Cx = test1.images.cov
I_in = np.diag(np.repeat(input_noise,Cx.shape[0]))
I_out = np.diag(np.repeat(output_noise,Cx.shape[0]))
W = test1.w_flat

np.matmul()


shape = shapes.get_shape_module("difference-of-gaussian")(torch.tensor(18, device = device), 1, torch.tensor(100, device = device)).to(device)
init_params = shape.shape_params
fun = test1.DoG_fit_func(shape, test1.kernel_centers)
yas = scipy.optimize.minimize(fun, init_params.detach().cpu().numpy(), method = "Nelder-Mead")