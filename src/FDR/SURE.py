import numpy as np
import torch
from scipy.optimize import minimize
import pywt
from .primaldual_multi_scaled_tune import make_solver
from functools import partial
from ray import tune
from .utils import *
from scipy.stats import beta
from ray.util.accelerators import NVIDIA_TESLA_V100
from ray.tune import CLIReporter, JupyterNotebookReporter
import sys
from ray.air.config import RunConfig

from ray.train import CheckpointConfig

checkpoint_config = CheckpointConfig( 
    checkpoint_frequency=0,  # Disable periodic checkpoints
    checkpoint_at_end=False  # Do not save a checkpoint at the end
)


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
    
    mad = np.median(np.abs(wavs - np.median(wavs)))
    
    sigma = mad / 0.6745
    
    return sigma**2
    
    
def tune_func(config, tol, eps, f, repeats, level, grid_y, sigma_sq, R,
              solver_options=None, probe_seed=None):
    #tune.utils.wait_for_gpu(target_util = 0.1, retry = 100)
    if torch.cuda.is_available(): # cuda gpus
        device_id = torch.cuda.current_device() 
        device = torch.device("cuda:{}".format(device_id)) 
        torch.cuda.set_device(device)

    elif torch.backends.mps.is_available(): # mac gpus
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        

    f, repeats, level, lmbda, nu, tol = \
        arraysToTensors(grid_y, repeats, level, 0, 0, tol, device)
    b = make_probes(f, R, probe_seed)
    
    theta = np.array([config['lmbda'], config['nu']])

    score = SURE_objective_tune(theta, tol=tol, eps=eps, f=f, repeats=repeats, 
                    level=level, grid_y=grid_y, sigma_sq=sigma_sq, b=b, R=R, solver_options=solver_options)
    return {'score' : score}


def make_probes(f, count, seed):
    generator = torch.Generator(device="cpu")
    if seed is None:
        seed = torch.randint(2**31-1, (), device="cpu").item()
    generator.manual_seed(int(seed))
    return torch.randn(list(f.shape)+[count], device="cpu", dtype=f.dtype, generator=generator).to(f.device)

def custom_loguniform(lower=0.001, upper=50, alpha=1.5, beta_b=1, size = 100):
    val = beta.rvs(alpha, beta_b, size=size)
    scaled_val = lower * ((upper/lower) ** val)
    return scaled_val

    
def SURE(model, maxiter = 100, R = 1, tuner = False, eps = 0.01, 
         wavelet = "db1", num_cpus = 4, num_gpus = 1, num_samples = 200, 
         nu_min = 0.001, nu_max=1, lmbda_min=1, lmbda_max=500, probe_seed=None):
    """ 

    Args:
        model (FDR class): 
        maxiter (int, optional): maximum number of iterations per model run. Defaults to 100.
        R (int, optional): number of times to compute model per iteration for Monte-Carlo averaged SURE. Defaults to 1 (no averaging).
        tuner (bool, optional): _description_. Defaults to False.
        eps (float, optional): step size to use for Monte-Carlo divergence computation (delta in paper). Defaults to 0.01.
        wavelet (str, optional): type of wavelet to use for computation of error variance. Defaults to "db1".
        num_cpus (int, optional): number of CPUs to assign to each parallel computation. Defaults to 4.
        num_gpus (int, optional): number of GPUs to assign to each parallel computation. Allows for fractional GPU usage (e.g. 0.25). Defaults to 1.
        num_samples (int, optional): number of hyperparameter combinations to compute the SURE for. Defaults to 200.
        nu_min (float, optional): lower bound on the boundary regularity parameter. Defaults to 0.001.
        nu_max (int, optional): upper bound on the boundary regularity parameter. Defaults to 10.
        lmbda_min (int, optional): lower bound on the data term parameter. Defaults to 1.
        lmbda_max (int, optional): upper bound on the data term parameter. Defaults to 500.

    Returns:
        Ray ResultGrid with tuner=True, or SciPy OptimizeResult for the bounded local search.
        probe_seed fixes the finite-difference perturbations shared across candidates.
    """

    if R < 1 or eps <= 0 or not (0 <= nu_min <= nu_max and 0 <= lmbda_min <= lmbda_max):
        raise ValueError("Provide positive probe count and epsilon and ordered nonnegative search bounds")
    sigma_sq = waveletDenoising(y=model.grid_y, wavelet=wavelet)
    solver_options = dict(scaled=model.scaled, scripted=model.scripted,
                          legacy_jump_projection=getattr(model, "legacy_jump_projection", True),
                          step_safety=getattr(model, "step_safety", 1.0),
                          fused_intervals=getattr(model, "fused_intervals", False))
    if probe_seed is None:
        probe_seed = torch.randint(2**31-1, (), device="cpu").item()

    if not tuner:
        f, repeats, levels, _, _, tol = model.arraysToTensors(
            model.grid_y, model.iter, model.level, model.lmbda, model.nu, model.tol)
        probes = make_probes(f, R, probe_seed)
        initial = [np.clip(model.lmbda,lmbda_min,lmbda_max), np.clip(model.nu,nu_min,nu_max)]
        # The half-level readout is piecewise smooth. Use a bounded
        # derivative-free search with identical probes at every evaluation.
        return minimize(SURE_objective_tune, initial,
                        args=(tol, eps, f, repeats, levels, model.grid_y, sigma_sq, probes, R, solver_options),
                        bounds=((lmbda_min,lmbda_max),(nu_min,nu_max)), method="Powell",
                        options={"maxiter": maxiter})
    elif tuner:
    #     search_space = {
    #     "lmbda": tune.uniform(1, 2e2),
    #     "nu": tune.sample_from(lambda spec:   lower * ((nu_max/lower) ** beta.rvs(0.5, 1))),
    # }

        search_space={
            # A random function
            "lmbda": tune.uniform(lmbda_min, lmbda_max),
            "nu":  tune.uniform(nu_min, nu_max)
            # Use the `spec.config` namespace to access other hyperparameters
            #"nu":
        }
        trainable_with_resources = tune.with_resources(
            partial(tune_func, tol=model.tol, eps=eps, f=model.grid_y, repeats=model.iter, 
                    level=model.level, grid_y=model.grid_y, sigma_sq=sigma_sq, R=R,
                    solver_options=solver_options, probe_seed=probe_seed),
            {"cpu": num_cpus, "gpu": num_gpus}
        )
                # Start the Ray Tune run
        analysis = tune.Tuner(
            trainable_with_resources,
            param_space=search_space,
            tune_config=tune.TuneConfig(num_samples=num_samples),  # number of different hyperparameter combinations to try
            run_config=RunConfig(progress_reporter=get_reporter())
        )

        # Get the hyperparameters of the best trial
        res = analysis.fit() #get_best_trial("objective", "min", "last")

    return res


def get_reporter(max_progress_rows=10, metric_column="custom_metric"):
    # Check if ipykernel is loaded
    is_jupyter = 'ipykernel' in sys.modules

    if is_jupyter:
        reporter = JupyterNotebookReporter(overwrite=True)
    else:
        reporter = CLIReporter(max_progress_rows=max_progress_rows)

    reporter.add_metric_column(metric_column)

    return reporter


def SURE_objective_tune(theta, tol, eps, f, repeats, level, grid_y, sigma_sq, b, R, solver_options=None):
    """Cell-mean MC-SURE with the same FDR solver for the base and each perturbation."""
    if R < 1 or eps <= 0 or tuple(b.shape) != tuple(f.shape)+(R,):
        raise ValueError("Supply one perturbation grid per probe and a positive epsilon")
    device = f.device
    solver = make_solver(device, **(solver_options or {}))
    lam, nu = (torch.as_tensor(value, device=device, dtype=f.dtype) for value in theta)
    levels = int(level)

    def fit(grid):
        lift = solver(grid, repeats, level, lam, nu, tol)[0]
        return isosurface(lift.detach().cpu().numpy(), levels, grid_y).ravel()

    # Keep all operations on the caller's stream. The readout synchronizes
    # with the device before NumPy evaluates the squared loss and divergence.
    with torch.no_grad():
        base = fit(f)
        squared_error = np.mean((np.asarray(grid_y).ravel()-base)**2)
        scores = []
        for i in range(R):
            probe = b[..., i].to(device=device, dtype=f.dtype)
            perturbed = fit((f+eps*probe).clamp(0,1))
            divergence = np.vdot(probe.cpu().numpy().ravel(), perturbed-base).real / eps
            scores.append(squared_error-sigma_sq+2*sigma_sq*divergence/base.size)
    return float(np.mean(scores))
