from FDD import FDD
from FDD.SURE import SURE
import numpy as np
import pandas as pd
import torch 
from matplotlib import pyplot as plt
import matplotlib as mpl
import ray
import boto3
import os
import pickle
from tabulate import tabulate
import random



def ft(x, jumps):
    temp = x**2 + np.sin(10 * x)  # change 10 to any other value to adjust the frequency of oscillation
    for (xj, jj) in jumps:
        if x > xj:
            temp += jj
    return temp

# Redefine the function to generate data with larger jumps, with last one going down
def generate1D(jumps=[(0.2013934, 0.6), (0.4023231, 1), (0.590349, 1.5), (0.7893434, -2)], sigma=0.1, N=500, data=None):
    if data is None:
        data = np.random.rand(N)  # draw N 1D points from a uniform

    # now sample the function values on the data points
    grid_f = np.zeros(data.shape)
    for i in range(data.shape[0]):
        grid_f[i] = ft(data[i], jumps)

    # Normalize function values to [0, 1]
    grid_f = (grid_f - np.min(grid_f)) / np.max(grid_f)  # - np.min(grid_f))

    # now add noise
    grid_sample = grid_f + np.random.normal(loc=0, scale=sigma, size=data.shape)  # add random Gaussian noise
    
    # now cast this data into a standard data format
    X = data.copy()
    Y = grid_sample.copy()
    u = grid_f.copy()

    return (X, Y, u)

def getOriginalImage(model, jumplocs = [0.2013934, 0.4023231, 0.590349, 0.7893434]):
    data = model.grid_x * np.max(model.X_raw, axis = 0) + np.min(model.X_raw, axis = 0) # draw 1000 2D points from a uniform
    data = data.squeeze()
    # original boundary
    
    temp_bdy = np.zeros_like(data)
    for jump in jumplocs:
        temp_bdy += ((data+0.5*model.resolution) < jump) * ((np.append(data[1:], 0) + 1.5*model.resolution) > jump)

    x, y, u = generate1D(N = data.size, data=data)


    return u, temp_bdy

if __name__ == "__main__":

    #----------------
    # parameters
    #----------------
    sigma=0.05
    S = 32
    N = 3000
    lmbda = 100
    nu = 0.002
    main_dir = "s3://projects-fdd/"
    data_out = os.path.join(main_dir, "data", "out")
    figs_dir = os.path.join(main_dir, "results", "figs")
    tabs_dir = os.path.join(main_dir, "results", "tabs")

    df = pd.read_csv(os.path.join(data_out, "simulations", "2022-08-02", "simulations_1d_sigma_0.05.csv"))
    
    N = df['N'].max()
    lmbda, nu, sigma, S = df[['lambda', 'nu', 'sigma', 'S']].loc[0]
    
    np.random.seed(3340)
    # Generate data with reduced noise
    X, Y, U = generate1D(sigma=0.05, N=N)

    resolution = 1/int(0.05*Y.size)
    model = FDD(Y, X, level = S, lmbda = lmbda, nu = nu, iter = 10000, tol = 5e-5, resolution=resolution,
        pick_nu = "MS", scaled = True, scripted = False, CI=False)

    results = model.run()
    u = results['u']
    fn = "1d_u.npy"
    np.save(fn, u)
    
    s3 = boto3.client('s3')
    with open(fn, "rb") as f:
        s3.upload_fileobj(f, "projects-fdd", "data/out/subsampling/" + fn)
        
    (test, b) = model.subSampling(nboot = 500)    
    test = np.stack(test, axis=0)
    fn = "1d_boots.npy"
    np.save(fn, test)
    with open(fn, "rb") as f:
        s3.upload_fileobj(f, "projects-fdd", "data/out/subsampling/" + fn)
        
    fn = "1d_b.npy"
    np.save(fn, b)
    with open(fn, "rb") as f:
        s3.upload_fileobj(f, "projects-fdd", "data/out/subsampling/" + fn)
        
    