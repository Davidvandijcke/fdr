import numpy as np
from FDD import FDD
from FDD.SURE import *
import ray
import numpy as np
import pandas as pd

def f(x,y, jsize):
  temp = np.sqrt((x-1/2)**2 + (y-1/2)**2)
  if temp < 1/4:
      return temp
  else:
      return temp + jsize

def generate2D(jsize=0.1, sigma = 0.02, N=500):
  data = np.random.rand(N, 2) # draw 1000 2D points from a uniform

  # now sample the function values on the data points
  grid_sample = np.zeros((data.shape[0],1))
  grid_f = np.zeros((data.shape[0],1))
  for i in range(data.shape[0]):
      grid_f[i] = f(data[i,0], data[i,1], jsize)
      grid_sample[i] = grid_f[i] + np.random.normal(loc = 0, scale = sigma) # add random Gaussian noise
  
  # now cast this data into a standard data format
  X = data.copy()
  Y = grid_sample.copy().flatten()
  u = grid_f.copy().flatten()

  return (X,Y,u)

@ray.remote(num_gpus=1)  # This decorator indicates that this function will be distributed, with each task using one GPU.
def train(config, jsize, sigma, N, lmbda, nu, S):
    # Here we randomly generate training data.
    X, Y, U = generate2D(jsize=jsize, sigma=sigma, N=N)

    if torch.cuda.is_available(): # cuda gpus
        device_id = torch.cuda.current_device() 
        device = torch.device("cuda:{}".format(device_id)) 
        torch.cuda.set_device(device)

    elif torch.backends.mps.is_available(): # mac gpus
        device = torch.device("mps")
        
    resolution = 1/int(np.sqrt(N*2/3))
    model = FDD(Y, X, level = S, lmbda = lmbda, nu = nu, iter = 5000, tol = 5e-5, resolution=resolution,
            pick_nu = "MS", scaled = True, scripted = False)
    
    u, jumps, J_grid, nrj, eps, it = model.run()
    mse = np.mean((u.flatten() - U.flatten())**2)
    temp = pd.DataFrame(jumps)
    temp[['alpha', 'N', 'S', 's', 'sigma', 'mse']] = jsize, N, S, sigma, mse
    return temp

configurations = [{"lr": 1} for _ in range(100)]  # Generate configurations for 10 models.

jsize = 0.15
sigma = 0.02
N = 500
lmbda = 10
nu = 0.01
S = 16
results = ray.get([train.remote(config, jsize, sigma, N, lmbda, nu, S) for config in configurations])
