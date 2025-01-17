import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from tempfile import gettempdir
from typing import Optional

import fire
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from data import KyotoNaturalImages
from model import RetinaVAE, OutputTerms, OutputMetrics
from util import cycle, kernel_images, flip_images, check_d
from analysis_utils import find_last_cp
import random as rnd


def set_seed(seed=None, seed_torch=True):
    if seed is None:
        seed = np.random.choice(2 ** 32)
        random.seed(seed)
        np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    print(f'Random seed {seed} has been set.')


def train(logdir: str = datetime.now().strftime(f"{gettempdir()}/%y%m%d-%H%M%S"),
          iterations: int = 1_000_000,
          #iterations: int = 3,
          batch_size: int = 128,
          data: str = "imagenet",
          kernel_size: int = 18,
          circle_masking: bool = True,
          neurons: int = 100,  # number of neurons, J
          jittering_start: Optional[int] = 0, #originally 200000
          jittering_stop: Optional[int] = 0, #originally 500000
          jittering_interval: int = 5000,
          jittering_power: float = 0.25,
          centering_weight: float = 0.02,
          centering_start: Optional[int] = 0, #originally 200000
          centering_stop: Optional[int] = 0, #originally 500000
          input_noise: float = 0.4,
          output_noise: float = 3,
          nonlinearity: str = "softplus",
          beta: float = -0.5,
          n_colors = 1,
          shape: Optional[str] = None, #'difference-of-gaussian', # "difference-of-gaussian" for Oneshape case #BUG: Can't use color 1 with "difference-of-gaussian"
          individual_shapes: bool = True,  # individual size of the RFs can be different for the Oneshape case
          optimizer: str = "sgd",  # can be "adam"
          learning_rate: float = 0.01, #Consider having a high learning rate at first then lower it. Pytorch has packages for this 
          rho: float = 1,
          maxgradnorm: float = 20.0,
          load_checkpoint: str = None,  # checkpoint file to resume training from
          fix_centers: bool = False,  # used if we want to fix the kernel_centers to learn params
          n_mosaics = 1, #Number of mosaics. Only relevant is fix_centers = True
          whiten_pca_ratio = None, #Default is None
          device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
          firing_restriction = "Lagrange",
          corr_noise_sd = 0, #Default is 0. Correlated input noise across space 
          image_restriction = "True", #Default is "True" 
          flip_odds = 0, #Flips L and S channels with a certain probability. Only works with 2 colors
          norm_image = False, #Removes the mean from each small image
          normalize_color = False):  #On the big images, normalize each color separately (mean and sd) instead of together.  

    train_args = deepcopy(locals())  # keep all arguments as a dictionary
    for arg in sys.argv:
        if arg.startswith("--") and arg[2:] not in train_args:
            raise ValueError(f"Unknown argument: {arg}")

    #SEED = 50
    #set_seed(seed=SEED)

    print(f"Logging to {logdir}")
    

    dataset = KyotoNaturalImages('kyoto_natim', kernel_size, circle_masking, device, n_colors, normalize_color, restriction = image_restriction, remove_mean = norm_image)
    if whiten_pca_ratio is not None:
        dataset.pca_color()
        dataset.whiten_pca(whiten_pca_ratio)
    data_covariance = dataset.covariance()
    data_loader = DataLoader(dataset, batch_size)
    data_iterator = cycle(data_loader)
    model_args = dict(
        kernel_size=kernel_size,
        neurons=neurons,
        input_noise=input_noise,
        output_noise=output_noise,
        nonlinearity=nonlinearity,
        shape = shape,
        individual_shapes = individual_shapes,
        data_covariance=data_covariance,
        beta=beta,
        rho=rho,
        fix_centers = fix_centers,
        n_colors = n_colors,
        n_mosaics = n_mosaics,
        corr_noise_sd = corr_noise_sd
    )
    
    model = RetinaVAE(**model_args).to(device)
    model_args["data_covariance"] = None
    model_args["sigma_prime"] = None

    #H_X = data_covariance.cholesky().diag().log2().sum().item() + model.D / 2.0 * np.log2(2 * np.pi * np.e)
    #print(f"H(X) = {H_X:.3f}")

    optimizer_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer]
    optimizer_kwargs_MI = dict(lr=learning_rate)
    optimizer_MI = optimizer_class(model.parameters(), **optimizer_kwargs_MI)
    
    
    
    all_params = [p for n, p in model.named_parameters() if p.requires_grad]
    last_iteration = 0
    if load_checkpoint is not None:
        path = "../../saves/" + load_checkpoint + "/"
        last_cp = find_last_cp(path)
        checkpoint = torch.load(path + last_cp)
        logdir = path
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Pickled model no longer supported for loading")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_MI.load_state_dict(checkpoint["optimizer_state_dict"])
        last_iteration = checkpoint["iteration"]
    if jittering_start is not None and jittering_stop is not None:
        assert jittering_interval > 0
        jittering_iterations = set(range(jittering_start, jittering_stop+1, jittering_interval))
    else:
        jittering_iterations = []

    writer = SummaryWriter(log_dir=logdir)
    writer.add_text("train_args", json.dumps(train_args))
    kernel_norm_penalty = 0
    #h_exp = torch.ones([neurons], device = "cuda")
    patience = 0
    cooldown_timer = 0
    
    model.train()
    C_z_estimate = torch.zeros([neurons, neurons], device = 'cpu') #Crashes if saved on gpu
    C_zx_estimate = torch.zeros([neurons,neurons], device = 'cpu')
    global gradS_all
    global gradL_all
    global h_all
    global dL_all
    global dS_all
    gradL_all = []
    gradS_all = []
    h_all = []
    dL_all = []
    dS_all = []
    with trange(last_iteration + 1, iterations + 1, ncols=99) as loop:
        
        for iteration in loop:
            
            if iteration % 1000 == 0 or iteration == 1:
                record_C = True
            else:
                record_C = False
                
            if iteration in jittering_iterations:
                model.encoder.jitter_kernels(jittering_power)
            
            batch = next(data_iterator).to(device)
            
            #THERE IS A BIG BUG WITH FLIP_ODDS. THE FLIP DOESN'T HAPPEN ON THE COV MATIRX 
            if flip_odds > 0:
                batch = flip_images(batch, flip_odds, device)
            
            torch.manual_seed(iteration)

            output: OutputTerms = model(batch, firing_restriction, corr_noise_sd, record_C = record_C) #This is the line where forward gets called
            metrics: OutputMetrics = output.calculate_metrics(iteration, firing_restriction)
            h_current = metrics.return_h().detach()
            effective_count = neurons
            
            loss_MI = metrics.final_loss(firing_restriction) + kernel_norm_penalty
                
                
            loss_FR = torch.sum(metrics.return_h()**2)
            kernel_variance = model.encoder.kernel_variance()
            
            if centering_start <= iteration < centering_stop:
                loss_MI = loss_MI + centering_weight * kernel_variance.mean()
            
            optimizer_MI.zero_grad()
        
            loss_MI.backward(retain_graph=True) 
            param_norm = torch.cat([param.data.flatten() for param in model.parameters()]).norm()
            grad_norm = torch.cat(
                [param.grad.data.flatten() for param in model.parameters() if param.grad is not None]).norm()
            for param in model.parameters():
                if param.shape == torch.Size([8,300]):
                    nnum = 109
                    #print('Printing this epoch')
                    dL = param[3,nnum].item()
                    dS = param[7,nnum].item()
                    #print('Weights: ', param[3,nnum].item(), param[7,nnum].item())
                    #print('Weights v2', model.encoder.shape_function.d[:,nnum])
                    gradL = param.grad.data[3,nnum].item()
                    gradS = param.grad.data[7,nnum].item()
                    #print('Gradients: ',gradL, gradS)
                    batch_means = np.mean(np.mean(batch.detach().cpu().numpy(), axis = 2), axis = 2)[0]
                    #print('Batch means: ', batch_means, np.mean(batch_means), batch_means[0] - batch_means[1])
                    #print(h_current[nnum] + 1)
                    h_all.append(h_current[nnum].item() + 1)
                    gradL_all.append(gradL)
                    gradS_all.append(gradS)
                    
                    dL_all.append(dL)
                    dS_all.append(dS)
                    
            
                    
            if firing_restriction == "Lagrange":
                model.Lambda.grad.neg_()

            if maxgradnorm:
                torch.nn.utils.clip_grad_norm_(all_params, maxgradnorm)
            
            optimizer_MI.step()
            
            #check_d(model, 109, 2, 600)
            
            model.encoder.normalize()
            
            
            
            
            loop.set_postfix(dict(
                KL=metrics.KL.mean().item(),
                loss=loss_MI.item()
            )) 
            
            C_z_estimate += model.encoder.C_z.detach().cpu()
            C_zx_estimate += model.encoder.C_zx.detach().cpu()
            
            if iteration % 10 == 0:                
                for key, value in output.__dict__.items():
                    if torch.is_tensor(value):
                        writer.add_scalar(f"terms/{key}", value.mean().item(), iteration)
                for key, value in metrics.__dict__.items():
                    if torch.is_tensor(value):
                        writer.add_scalar(f"terms/{key}", value.mean().item(), iteration)
                #writer.add_scalar(f"terms/MI", H_X - metrics.loss.mean().item(), iteration)
                writer.add_scalar(f"train/grad_norm", grad_norm.item(), iteration)
                writer.add_scalar(f"train/param_norm", param_norm.item(), iteration)
                writer.add_scalar(f"train/kernel_variance", kernel_variance.item(), iteration)
                writer.add_scalar(f"train/final_loss", loss_MI.item(), iteration)
                writer.add_scalar(f"train/Firing_rate_loss", loss_FR.item(), iteration)
                if torch.is_tensor(kernel_norm_penalty):
                    writer.add_scalar(f"train/kernel_norm_penalty", kernel_norm_penalty.item(), iteration)

            if iteration % 1000 == 0 or iteration == 1:
                W = model.encoder.W.detach().cpu().numpy()
                for param in model.parameters():
                    
                    if (np.array(param.shape) == [kernel_size*kernel_size*n_colors, neurons]).all():
                        W_grad = param.grad.detach().cpu().numpy()
                        
                writer.add_image('kernels', kernel_images(W, kernel_size, image_channels = model.encoder.image_channels), iteration)
                if shape is None:
                    writer.add_image('W_grad', kernel_images(W_grad, kernel_size, image_channels = model.encoder.image_channels), iteration)

                writer.add_image('MI_numerator', C_z_estimate/1000, iteration, dataformats="HW")
                writer.add_image('MI_denominator', C_zx_estimate/1000, iteration, dataformats="HW")
                writer.add_image('WCxW', model.encoder.WCxW, iteration, dataformats = "HW")
                
                C_z_estimate = torch.zeros(C_z_estimate.shape, device = 'cpu')
                C_zx_estimate = torch.zeros(C_zx_estimate.shape, device = 'cpu')
                Lambda = model.Lambda.detach().cpu().numpy()
                try:
                    writer.add_histogram("histograms/λ", Lambda, iteration, bins=100)
                except ValueError:
                    print(Lambda, np.max(W), np.min(W))
                     
                r = output.r.detach().cpu().numpy().mean(-1)  
                writer.add_histogram("histogram/r", r, iteration, bins=100)

                #gain = model.encoder.logA.detach().exp().cpu().numpy()
                #writer.add_histogram("histogram/gain", gain, iteration, bins=100)
                bias = model.encoder.logB.detach().exp().cpu().numpy()
                writer.add_histogram("histogram/bias", bias, iteration, bins=100)
                if hasattr(model.encoder, "shape_function"):
                    if hasattr(model.encoder.shape_function, "a"):
                        writer.add_histogram("histogram/diffgaussian_a", model.encoder.shape_function.a, iteration, bins=100)
                    if hasattr(model.encoder.shape_function, "b"):
                        writer.add_histogram("histogram/diffgaussian_b", model.encoder.shape_function.b, iteration, bins=100)
                    if hasattr(model.encoder.shape_function, "c"):
                        writer.add_histogram("histogram/diffgaussian_c", model.encoder.shape_function.c, iteration, bins=100)
                    if hasattr(model.encoder.shape_function, "d"):
                        writer.add_histogram("histogram/diffgaussian_d", model.encoder.shape_function.d, iteration, bins=100)
            
            if iteration % 1000 == 0:
                to_ignore = ["data_covariance", "sigma_prime"]
                to_restore = {}
                for key, value in model.encoder._buffers.items():
                    if key in to_ignore:
                        to_restore[key] = value
                        model.encoder._buffers[key] = None
                if load_checkpoint is None:
                    torch.save(model, f"{logdir}/model-{iteration}.pt")
                    cp_save = f"{logdir}/checkpoint-{iteration}.pt"
                else:
                    torch.save(model, path + f"model-{iteration}.pt")
                    cp_save = path + f"checkpoint-{iteration}.pt"
                for key, value in to_restore.items():
                    model.encoder.register_buffer(key, value, persistent=False)
                
                MI_matrices = torch.stack([C_z_estimate, C_zx_estimate])
                
                torch.save(dict(
                    iteration=iteration,
                    args=train_args,
                    model_args=model_args,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer_MI.state_dict(),
                    weights= model.encoder.W,
                    MI_matrices = MI_matrices,
                    
                    
                ), cp_save)
            
            writer.flush()
    

    writer.close()


if __name__ == "__main__":
    fire.Fire(train)
