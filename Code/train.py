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
from util import cycle, kernel_images
from analysis_utils import find_last_cp


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
          iterations: int = 2_000_000,
          #iterations: int = 3,
          batch_size: int = 64,
          data: str = "imagenet",
          kernel_size: int = 12,
          circle_masking: bool = True,
          dog_prior: bool = False,
          neurons: int = 498,  # number of neurons, J
          jittering_start: Optional[int] = None, #originally 200000
          jittering_stop: Optional[int] = None, #originally 500000
          jittering_interval: int = 5000,
          jittering_power: float = 0.25,
          centering_weight: float = 0.02,
          centering_start: Optional[int] = None, #originally 200000
          centering_stop: Optional[int] = None, #originally 500000
          input_noise: float = 0.4,
          output_noise: float = 3.0,
          nonlinearity: str = "softplus",
          beta: float = -0.5,
          n_colors = 3,
          shape: Optional[str] = "difference-of-gaussian", # "difference-of-gaussian" for Oneshape case
          individual_shapes: bool = True,  # individual size of the RFs can be different for the Oneshape case
          optimizer: str = "adam",  # can be "adam"
          learning_rate: float = 0.001, #Consider having a high learning rate at first then lower it. Pytorch has packages for this 
          rho: float = 1,
          maxgradnorm: float = 20.0,
          load_checkpoint: str = None, #"230705-141246",  # checkpoint file to resume training from
          fix_centers: bool = True,  # used if we want to fix the kernel_centers to learn params
          n_mosaics = 6,
          whiten_pca_ratio = None,
          device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

    train_args = deepcopy(locals())  # keep all arguments as a dictionary
    for arg in sys.argv:
        if arg.startswith("--") and arg[2:] not in train_args:
            raise ValueError(f"Unknown argument: {arg}")

    #SEED = 50
    #set_seed(seed=SEED)

    print(f"Logging to {logdir}")
    writer = SummaryWriter(log_dir=logdir)
    writer.add_text("train_args", json.dumps(train_args))

    dataset = KyotoNaturalImages('kyoto_natim', kernel_size, circle_masking, device, n_colors)
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
        n_mosaics = n_mosaics
    )
     
    model = RetinaVAE(**model_args).to(device)
    model_args["data_covariance"] = None
    model_args["sigma_prime"] = None

    H_X = data_covariance.cholesky().diag().log2().sum().item() + model.D / 2.0 * np.log2(2 * np.pi * np.e)
    print(f"H(X) = {H_X:.3f}")

    optimizer_class = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}[optimizer]
    optimizer_kwargs = dict(lr=learning_rate)

    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
    all_params = [p for n, p in model.named_parameters() if p.requires_grad]
    last_iteration = 0
    if load_checkpoint is not None:
        path = "saves/" + load_checkpoint + "/"
        last_cp = find_last_cp(path)
        checkpoint = torch.load(path + last_cp)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Pickled model no longer supported for loading")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        last_iteration = checkpoint["iteration"]
    if jittering_start is not None and jittering_stop is not None:
        assert jittering_interval > 0
        jittering_iterations = set(range(jittering_start, jittering_stop+1, jittering_interval))
    else:
        jittering_iterations = []

    kernel_norm_penalty = 0

    model.train()
    with trange(last_iteration + 1, iterations + 1, ncols=99) as loop:
        for iteration in loop:
            
            if iteration in jittering_iterations:
                model.encoder.jitter_kernels(jittering_power)
            
            batch = next(data_iterator).to(device)
            #print('batch', batch.shape)
            
            torch.manual_seed(iteration)
            #print('pre', batch.shape) #for debugging
            output: OutputTerms = model(batch) #This is the line where forward gets called
            #print('post') #for debugging
            metrics: OutputMetrics = output.calculate_metrics(iteration)
            
            effective_count = neurons
            
            loss = metrics.final_loss() + kernel_norm_penalty
            kernel_variance = model.encoder.kernel_variance()
            
            if not fix_centers:
                if centering_start <= iteration < centering_stop:
                    loss = loss + centering_weight * kernel_variance.mean()
            
            optimizer.zero_grad()
            loss.backward()
            param_norm = torch.cat([param.data.flatten() for param in model.parameters()]).norm()
            grad_norm = torch.cat(
                [param.grad.data.flatten() for param in model.parameters() if param.grad is not None]).norm()
            model.Lambda.grad.neg_()

            if maxgradnorm:
                torch.nn.utils.clip_grad_norm_(all_params, maxgradnorm)
            
            optimizer.step()
            model.encoder.normalize()
            
            loop.set_postfix(dict(
                KL=metrics.KL.mean().item(),
                loss=loss.item()
            ))
            
            if iteration % 10 == 0:
                for key, value in output.__dict__.items():
                    if torch.is_tensor(value):
                        writer.add_scalar(f"terms/{key}", value.mean().item(), iteration)
                for key, value in metrics.__dict__.items():
                    if torch.is_tensor(value):
                        writer.add_scalar(f"terms/{key}", value.mean().item(), iteration)
                writer.add_scalar(f"terms/MI", H_X - metrics.loss.mean().item(), iteration)
                writer.add_scalar(f"train/grad_norm", grad_norm.item(), iteration)
                writer.add_scalar(f"train/param_norm", param_norm.item(), iteration)
                writer.add_scalar(f"train/kernel_variance", kernel_variance.item(), iteration)
                writer.add_scalar(f"train/final_loss", loss.item(), iteration)
                if torch.is_tensor(kernel_norm_penalty):
                    writer.add_scalar(f"train/kernel_norm_penalty", kernel_norm_penalty.item(), iteration)

            if iteration % 1000 == 0 or iteration == 1:
                W = model.encoder.W.detach().cpu().numpy()
                writer.add_image('kernels', kernel_images(W, kernel_size, image_channels = model.encoder.image_channels), iteration) #David: I still have to change image_channels manually. I cannot figure out how to pass it from here (yet)

                Lambda = model.Lambda.detach().cpu().numpy()
                try:
                    writer.add_histogram("histograms/λ", Lambda, iteration, bins=100)
                except ValueError:
                    print(Lambda, np.max(W), np.min(W))
                    

                r = output.r.detach().cpu().numpy().mean(-1)
                writer.add_histogram("histogram/r", r, iteration, bins=100)

                gain = model.encoder.logA.detach().exp().cpu().numpy()
                writer.add_histogram("histogram/gain", gain, iteration, bins=100)

                bias = model.encoder.logB.detach().exp().cpu().numpy()
                writer.add_histogram("histogram/bias", bias, iteration, bins=100)
                if hasattr(model.encoder, "shape_function"):
                    if isinstance(model.encoder.shape_function, nn.ModuleList):
                        a = torch.cat([shape.a for shape in model.encoder.shape_function])
                    else:
                        a = model.encoder.shape_function.a
                    writer.add_histogram("histogram/diffgaussian_a", a, iteration, bins=100)
                    if hasattr(model.encoder.shape_function, "b"):
                        if isinstance(model.encoder.shape_function, nn.ModuleList):
                            b = torch.cat([shape.b for shape in model.encoder.shape_function])
                        else:
                            b = model.encoder.shape_function.b
                        writer.add_histogram("histogram/diffgaussian_b", b, iteration, bins=100)
                #if hasattr(model.encoder, "shape_function"):
                #    if isinstance(model.encoder.shape_function, HexagonalGrid):
                #        a = np.array([shape.a.item() for shape in model.encoder.shape_function.shape])
                #        writer.add_histogram("histogram/gaussian_a", a, iteration, bins=100)
                #    else:
                #        a = model.encoder.shape_function.a
                #        writer.add_histogram("histogram/diffgaussian_a", a, iteration, bins=100)
                #        if hasattr(model.encoder.shape_function, "b"):
                #            b = model.encoder.shape_function.b
                #            writer.add_histogram("histogram/diffgaussian_b", b, iteration, bins=100)
            
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
                
                torch.save(dict(
                    iteration=iteration,
                    args=train_args,
                    model_args=model_args,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    weights= model.encoder.W,
                ), cp_save)
            
            writer.flush()

    writer.close()


if __name__ == "__main__":
    fire.Fire(train)