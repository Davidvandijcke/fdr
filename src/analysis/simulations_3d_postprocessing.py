from FDD import FDD
from FDD.SURE import SURE
import numpy as np
import pandas as pd
import torch 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def f(x, y, z, jsize):
    temp = np.sqrt((x - 1/2)**2 + (y - 1/2)**2 + (z - 1/2)**2)
    if temp < 1/4:
        return temp
    else:
        return temp + jsize

def generate3D(jsize=0.1, sigma=0.02, N=500):
    data = np.random.rand(N, 3) # draw 1000 3D points from a uniform

    # now sample the function values on the data points
    grid_sample = np.zeros((data.shape[0],1))
    grid_f = np.zeros((data.shape[0],1))
    for i in range(data.shape[0]):
        grid_f[i] = f(data[i,0], data[i,1], data[i,2], jsize)
        grid_sample[i] = grid_f[i] + np.random.normal(loc=0, scale=sigma) # add random Gaussian noise

    # now cast this data into a standard data format
    X = data.copy()
    Y = grid_sample.copy().flatten()
    u = grid_f.copy().flatten()

    return (X, Y, u)

def plot3D(X, Y):
    
    # Create a new figure and add a 3D subplot to it
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the generated data. Color of each point is determined by its Y value
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap='coolwarm', alpha = 0.5, s=1)

    # Add a colorbar to the plot
    fig.colorbar(scatter)

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return (fig, ax)



if __name__ == "__main__":
    np.random.seed(0)
    import sys
    old_stdout = sys.stdout

    log_file = open("message.log","w")

    sys.stdout = log_file

    #-------------
    # parameters
    #-------------
    
    sigma = 0.05
    S = 32
    #----
    
    X, Y, U = generate3D(jsize = 0, sigma=sigma, N=10000)
    std = np.std(Y)
    jsize = 0.5 * std
    

    #u, jumps, J_grid, nrj, eps, it = model.run()
    
    fn = '3D_SURE.pkl'
    

    ffrom = f"'s3://ipsos-dvd/fdd/data/{fn}'"
    fto = f"'/Users/davidvandijcke/Dropbox (University of Michigan)/rdd/data/out/simulations/{fn}'"
    !aws s3 cp $ffrom $fto --profile ipsos
    
    with open(fto.replace("'",''), "rb") as f:
        res = pickle.load(f)

    best = res.get_best_result(metric = "score", mode = "min")

    config = best.metrics['config']
    lmbda, nu = config['lmbda'], config['nu']
    
    X, Y, U = generate3D(jsize = jsize, sigma=sigma, N=10000)

    N = Y.size
    resolution = 1/int((N*2/3)**(1/3))
    model = FDD(Y, X, level = S, lmbda = 1, nu = 0.01, iter = 5000, tol = 5e-6, resolution=resolution,
            pick_nu = "MS", scaled = True, scripted = False)

    u, jumps, J_grid, nrj, eps, it = model.run()
    
    # flatten all but last dimension of grid_x
    #grid_x = model.grid_x.reshape(-1, model.grid_x.shape[-1])
    
    #plot3D(grid_x, J_grid.flatten())
    
    