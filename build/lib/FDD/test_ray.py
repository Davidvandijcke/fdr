import numpy as np
from FDD import FDD
from FDD.SURE import *
import ray
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ray.util.accelerators import NVIDIA_TESLA_V100


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

@ray.remote(num_gpus=1, accelerator_type=NVIDIA_TESLA_V100)  # This decorator indicates that this function will be distributed, with each task using one GPU.
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

if __name__ == "__main__":
    # configurations = [{"lr": 1} for _ in range(100)]  # Generate configurations for 10 models.

    # jsize = 0.15
    # sigma = 0.02
    # N = 500
    # lmbda = 10
    # nu = 0.01
    # S = 16
    # results = ray.get([train.remote(config, jsize, sigma, N, lmbda, nu, S) for config in configurations])

    # check simulations
    df = pd.read_csv("/Users/davidvandijcke/Downloads/simulations_2d_sigma_0.01_jsize_0.035537513272984836.csv")

    # Group by 'alpha', 'N', and 'S' and calculate the mean 'Y_jumpsize'
    df['Y_jumpsize'] = df['Y_jumpsize'].abs()
    mean_jumpsize = df.groupby(['alpha', 'N', 'S', 's']).agg({'Y_jumpsize' : 'mean', 
                                                'mse' : 'mean', 
                                                'jump_neg' : 'mean',
                                                'jump_pos' : 'mean',
                                                'Y_jumpfrom' : 'mean',
                                                'Y_jumpto' : 'mean',
                                                'lambda' : 'mean', 
                                                'nu' : 'mean'}).reset_index()
    mean_jumpsize = mean_jumpsize.groupby(['alpha', 'N', 'S']).agg({'Y_jumpsize' : 'mean', 
                                                'mse' : 'mean', 
                                                'jump_neg' : 'mean',
                                                'jump_pos' : 'mean', 
                                                'Y_jumpfrom' : 'mean',
                                                'Y_jumpto' : 'mean',
                                                'lambda' : 'mean', 
                                                'nu' : 'mean'}).reset_index()

    # Create a new column 'N_S' that combines 'N' and 'S' as a tuple
    #?mean_jumpsize['N_S'] = mean_jumpsize.apply(lambda row: f"{row['N']}_{row['S']}", axis=1)

    # Create the pivot table with 'alpha' as rows and 'N_S' as columns
    pivot_table = mean_jumpsize.pivot_table(index='alpha', columns='N', values='Y_jumpsize')

    # Optional: sort the index and columns if needed
    pivot_table = pivot_table.sort_index(axis=0).sort_index(axis=1)

    # Display the pivot table
    print(pivot_table)
    
    ## test SURE
    # look at results
    jsize = 0.1
    sigma = 0.01
    N = 1000
    lmbda = 10
    nu = 0.01
    S = 16
    num_samples = 100
    R = 1
    num_gpus = 1
    num_cpus = 8
        
    
    
    X, Y, U = generate2D(jsize, sigma=0.01, N=N)
    resolution = 1/int(np.sqrt(N*2/3))


    model = FDD(Y, X, level = S, lmbda = 1, 
                nu = 0.1, iter = 10000, tol = 5e-5, 
                resolution=resolution, pick_nu = "MS", 
                scaled = True)

    res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
        num_gpus=num_gpus, num_cpus=num_cpus)
    best = res.get_best_result(metric = "score", mode = "min")

    config = best.metrics['config']
    lmbda, nu = config['lmbda'], config['nu']

    print(f"lambda {lmbda}, nu {nu}")
    
    # lmbda = 30.70987996835588
    # nu = 0.005081853584676989
    # model = FDD(Y, X, level = S, lmbda = lmbda, 
    #             nu = nu, iter = 10000, tol = 5e-5, 
    #             resolution=resolution, pick_nu = "MS", 
    #             scaled = True)
    # u, jumps, J_grid, nrj, eps, it = model.run()
    
    # def boundary(self, u):
        
    #     u_diff = self.forward_differences(u, D = len(u.shape))
    #     #u_diff = u_diff / self.resolution # scale FD by side length
    #     u_norm = np.linalg.norm(u_diff, axis = 0, ord = 2) # 2-norm

    #     if self.pick_nu == "kmeans":
    #         nu = self.pickKMeans(u_norm)
    #     else:
    #         nu = np.sqrt(self.nu)
        
    #     # find the boundary on the grid by comparing the gradient norm to the threshold
    #     J_grid = (u_norm >= nu).astype(int)
        
                
    #     # scale u back to get correct jump sizes
    #     if not self.image:
    #         u = u * np.max(self.Y_raw, axis = -1)
        
    #     ## find the boundary on the point cloud
    #     jumps = self.boundaryGridToData(J_grid, u)
        
    #     # test_grid = np.zeros(self.grid_y.shape)
    #     # for row in jumps:
    #     #     test_grid[tuple(row)[:-2]] = 1

    #     return (J_grid, jumps)
    
    # from types import MethodType

    # model.boundary = MethodType(boundary, model)

    
    # model.pick_nu = "MS"
    # J_grid, jumps = model.boundary(u)
    # temp = pd.DataFrame(jumps)

    # import inspect 
    # lines = inspect.getsource(model.boundary)
    # print(lines)
