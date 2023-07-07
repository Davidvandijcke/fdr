from FDD import FDD
from FDD.SURE import SURE
import numpy as np
import pandas as pd
import torch 
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from simulations_3d import *

def plot3D(X, Y, cmap = "coolwarm"):
    
    # Create a new figure and add a 3D subplot to it
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the generated data. Color of each point is determined by its Y value
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=cmap, alpha = 0.5, s=1)

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
    #!aws s3 cp $ffrom $fto --profile ipsos
    
    with open(fto.replace("'",''), "rb") as f:
        res = pickle.load(f)

    best = res.get_best_result(metric = "score", mode = "min")

    config = best.metrics['config']
    lmbda, nu = config['lmbda'], config['nu']
    
    lmbda = 100
    
    X, Y, U = generate3D(jsize = jsize, sigma=sigma, N=1000)

    N = Y.size
    resolution = 1/int((N*2/3)**(1/3))
    model = FDD(Y, X, level = S, lmbda = lmbda, nu = nu, iter = 10000, tol = 1e-5, resolution=resolution,
            pick_nu = "MS", scaled = True, scripted = False)

    u, jumps, J_grid, nrj, eps, it = model.run()
    
    temp = pd.DataFrame(jumps)
    
    # flatten all but last dimension of grid_x
    grid_x = model.grid_x.reshape(-1, model.grid_x.shape[-1])
    plot3D(grid_x, J_grid.flatten(), cmap = "binary")
    
    