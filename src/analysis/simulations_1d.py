from FDD import FDD
from FDD.SURE import SURE
import numpy as np
import pandas as pd
import torch 
from matplotlib import pyplot as plt
import ray
#import boto3
import os
import pickle

def f(x, jumps):
    temp = x**2 + np.sin(10 * x)  # change 10 to any other value to adjust the frequency of oscillation
    for (xj, jj) in jumps:
        if x > xj:
            temp += jj
    return temp

# Redefine the function to generate data with larger jumps, with last one going down
def generate1D(jumps=[(0.2, 0.9), (0.4, 1), (0.6, 1.5), (0.8, -2)], sigma=0.1, N=500):
    data = np.random.rand(N)  # draw N 1D points from a uniform

    # now sample the function values on the data points
    grid_f = np.zeros(data.shape)
    for i in range(data.shape[0]):
        grid_f[i] = f(data[i], jumps)

    # Normalize function values to [0, 1]
    grid_f = (grid_f - np.min(grid_f)) / (np.max(grid_f) - np.min(grid_f))

    # now add noise
    grid_sample = grid_f + np.random.normal(loc=0, scale=sigma, size=data.shape)  # add random Gaussian noise
    
    # now cast this data into a standard data format
    X = data.copy()
    Y = grid_sample.copy()
    u = grid_f.copy()

    return (X, Y, u)



if __name__ == "__main__":

    #----------------
    # parameters
    #----------------
    sigma=0.05
    S = 32
    N = 1000
    lmbda = 1000
    nu = 0.02
    num_samples = 400 # 225 #  400 # 400 # 400 # 200
    R =  3 # 3 # 3 # 3 # 5
    num_gpus = 1
    num_cpus = 2
    
    N_list = [100, 500, 1000, 10000]
    N_sure = max(N_list)
    S = 32
    num_samples = 400 #  400 # 400 # 200
    num_sims = 100 # 100 # 100 # 100
    R = 3 #  3 # 3 # 5
    num_gpus = 1
    num_cpus = 2
    fdate = "2022-06-23"

    @ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)  # This decorator indicates that this function will be distributed, with each task using one GPU.
    def train(config, jsize, sigma, N, lmbda, nu, S):
        # Here we randomly generate training data.
        X, Y, U = generate2D(jsize=jsize, sigma=sigma, N=N)
        
        #tune.utils.wait_for_gpu(target_util = 0.1, retry = 100000)


        if torch.cuda.is_available(): # cuda gpus
            device_id = torch.cuda.current_device() 
            device = torch.device("cuda:{}".format(device_id)) 
            torch.cuda.set_device(device)

        elif torch.backends.mps.is_available(): # mac gpus
            device = torch.device("mps")
            
        resolution = 1/int(np.sqrt(N*0.25))
        model = FDD(Y, X, level = S, lmbda = lmbda, nu = nu, iter = 100000, tol = 1e-5, resolution=resolution,
                pick_nu = "MS", scaled = True, scripted = False)
        
        u, jumps, J_grid, nrj, eps, it = model.run()
        
        u_original, J_original = getOriginalImage(model, jsize)

        
        mse = np.mean((u - u_original)**2)
        jump_pos = np.sum(J_grid * (1-J_original)) / np.sum(1-J_original) # false positive rate (significance)
        jump_neg = np.sum((1-J_grid) * (J_original)) / np.sum(J_original) # false negative rate (1-power)
        
        temp = pd.DataFrame(jumps)
        temp[['alpha', 'N', 'S', 's', 'sigma', 'lambda', 'nu', 'jump_neg', 
              'jump_pos', 'mse']] = jsize, N, S, config, sigma, lmbda, nu, jump_neg, jump_pos, mse
        return temp

    np.random.seed(34)
    # Generate data with reduced noise
    X, Y, u = generate1D(sigma=0.05, N=N)

    # Sort data for plotting
    sort_inds = np.argsort(X)
    X_sorted = X[sort_inds]
    Y_sorted = Y[sort_inds]
    u_sorted = u[sort_inds]


    resolution = 1/int(Y.size*0.25)
    model = FDD(Y, X, level = S, lmbda = lmbda, nu = nu, iter = 10000, tol = 5e-5, resolution=resolution,
        pick_nu = "MS", scaled = True, scripted = False)

    u, jumps, J_grid, nrj, eps, it = model.run()

    res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
        num_gpus=num_gpus, num_cpus=num_cpus)

    file_name = '1D_SURE.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(res, file)