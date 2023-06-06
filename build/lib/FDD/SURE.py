import numpy as np
import torch
from scipy.optimize import minimize
import pywt
from .primaldual_multi_scaled_tune import PrimalDual
from functools import partial
from ray import tune
from .utils import *
from scipy.stats import beta




def gridSearch(theta, args):
    lmbda_list = [1, 5, 10, 20, 50, 100, 300, 500]
    nu_list = [0.001, 0.005, 0.01, 0.02 ,0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

    objlist = []
    arglist = []
    for lmbda in lmbda_list:
        for nu in nu_list:
            # perform a grid search on the SURE_objective, retain all the values
            # and then pick the best one
            theta = np.array([lmbda, nu])
            objlist.append(SURE_objective_tune(theta, *args))
            arglist.append((lmbda, nu))
    
    # get index of minimum value
    min_idx = np.argmin(objlist)
    
    # get args of minimum value
    best_args = arglist[min_idx]
    
    return best_args
        
            
def waveletDenoising(y, wavelet : str = "db1"):
    
    coeffs = pywt.wavedecn(y.squeeze(), wavelet)
    
    # Get detail coefficients at the finest scale
    details = coeffs[-1]

    
    wavs = []  
    for key in details.keys():
        # Flatten the array to 1D for MAD calculation
        coeff_arr = np.ravel(details[key])
        wavs.append(coeff_arr)
    wavs = np.concatenate(wavs)
    
    mad = np.median(np.abs(wavs))
    
    sigma = mad / 0.6745
    
    return sigma**2
    
    
def tune_func(config, tol, eps, f, repeats, level, grid_y, sigma_sq, R):
    #tune.utils.wait_for_gpu(target_util = 0.1, retry = 100)
    if torch.cuda.is_available(): # cuda gpus
        device_id = torch.cuda.current_device() 
        device = torch.device("cuda:{}".format(device_id)) 
        torch.cuda.set_device(device)

    elif torch.backends.mps.is_available(): # mac gpus
        device = torch.device("mps")
        

    f, repeats, level, lmbda, nu, tol = \
        arraysToTensors(grid_y, repeats, level, 0, 0, tol, device)
    b = torch.randn(list(f.shape) + [R], device = device) 
    
    theta = np.array([config['lmbda'], config['nu']])

    score = SURE_objective_tune(theta, tol=tol, eps=eps, f=f, repeats=repeats, 
                    level=level, grid_y=grid_y, sigma_sq=sigma_sq, b=b, R=R)
    return {'score' : score}

def custom_loguniform(lower=0.001, upper=50, alpha=0.5, beta_b=1, size = 100):
    val = beta.rvs(alpha, beta_b, size=size)
    scaled_val = lower * ((upper/lower) ** val)
    return scaled_val

    
def SURE(model, maxiter = 100, R = 1, tuner = False, eps = 0.01, 
         wavelet = "db1", num_cpus = 4, num_gpus = 1, num_samples = 200):

    sigma_sq = waveletDenoising(y=model.grid_y, wavelet=wavelet)
    N = model.grid_y.size
    y = model.grid_y.squeeze()
    y_diff = model.forward_differences(y, D = len(y.shape)) / model.resolution
    y_norm = np.linalg.norm(y_diff, ord = 2, axis = 0)**2


    if model.scaled:
      nu_max = y_norm.max()
    else:
      nu_max = 1

    if not tuner:
        config = {'lmbda' : 1, 'nu' : 0.01}
        res = \
            minimize(SURE_objective_tune, config, 
                    tuple([model.tol, eps, model.grid_y, model.iter, model.level, model.grid_y, sigma_sq, R]),
                    method = "BFGS", tol = 1*10**(-9), 
                    options = {'disp' : True, 'maxiter' : maxiter}, bounds = ((1, 200), (0, nu_max)))
    elif tuner:
        lower = 0.001
    #     search_space = {
    #     "lmbda": tune.uniform(1, 2e2),
    #     "nu": tune.sample_from(lambda spec:   lower * ((nu_max/lower) ** beta.rvs(0.5, 1))),
    # }
        nu_grid = custom_loguniform(lower = lower, upper = nu_max, size = num_samples)
        search_space={
            # A random function
            "lmbda": tune.uniform(1, 2e2),
            "nu":  tune.choice(nu_grid)
            # Use the `spec.config` namespace to access other hyperparameters
            #"nu":
        }
        trainable_with_resources = tune.with_resources(
            partial(tune_func, tol=model.tol, eps=eps, f=model.grid_y, repeats=model.iter, 
                    level=model.level, grid_y=model.grid_y, sigma_sq=sigma_sq, R=R), 
            {"cpu": num_cpus, "gpu": num_gpus}
        )
                # Start the Ray Tune run
        analysis = tune.Tuner(
            trainable_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(num_samples=num_samples),  # number of different hyperparameter combinations to try
        )

        # Get the hyperparameters of the best trial
        res = analysis.fit() #get_best_trial("objective", "min", "last")

    return res





def SURE_objective_tune(theta, tol, eps, f, repeats, level, grid_y, sigma_sq, b, R):

    device = f.device

    sure = []
    
    lvl = level.cpu().detach().numpy()

    lmbda_torch = torch.tensor(theta[0], device = device, dtype = torch.float32)
    nu_torch = torch.tensor(theta[1], device = device, dtype = torch.float32)
    n = grid_y.size # flatten().shape[0]
    model = PrimalDual()
    v = model.forward(f, repeats, level, lmbda_torch, nu_torch, tol)[0]
    u = isosurface(v.cpu().detach().numpy(), lvl, grid_y)

    u_dist = np.mean(np.abs(grid_y.flatten() - u.flatten())**2)

    for r in range(R):

        bt = b[...,r]
        f_eps = f + bt * eps
        f_eps = torch.clamp(f_eps, min = 0, max = 1)

        v_eps = model.forward(f_eps, repeats, level, lmbda_torch, nu_torch, tol)[0]
        u_eps = isosurface(v_eps.cpu().detach().numpy(), lvl, grid_y)

        divf_y = np.real(np.vdot(bt.cpu().detach().numpy().squeeze().flatten(), 
                                u_eps.flatten() - u.flatten())) / (eps)
        sure.append(u_dist - sigma_sq + 2 * sigma_sq * divf_y / n)
        # TODO: should be euclidean norm
        sure = np.mean(sure)

    return sure