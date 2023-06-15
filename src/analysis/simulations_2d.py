from FDD import FDD
from FDD.SURE import SURE
import numpy as np
import pandas as pd
import torch 
from matplotlib import pyplot as plt
import ray
import boto3

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

def getOriginalImage(model, jsize):
    data = model.grid_x * np.max(model.X_raw, axis = 0) + np.min(model.X_raw, axis = 0) # draw 1000 2D points from a uniform
    
    # original boundary
    temp = np.sqrt((data[...,0]-1/2)**2 + (data[...,1]-1/2)**2) < 1/4
    temp = np.linalg.norm(model.forward_differences(temp, D = len(temp.shape)), axis = 0)
    temp_bdy = temp > 0
    
    u_original = np.zeros(data.shape[:-1]) 
    it = np.nditer(data[...,0], flags = ['multi_index'])
    for i in it:
        idx = it.multi_index
        x = data[idx][0]
        y = data[idx][1]
        u_original[idx] = f(x, y, jsize=jsize)
        
        
    return u_original, temp_bdy


if __name__ == "__main__":
    np.random.seed(0)
    import sys
    old_stdout = sys.stdout

    log_file = open("message.log","w")

    sys.stdout = log_file

    #-------------
    # parameters
    #-------------
    N_list = [100, 500, 1000, 10000]
    N_sure = max(N_list)
    S = 16
    num_samples = 2 #  400 # 400 # 200
    num_sims = 1 # 100 # 100 # 100
    R = 1 #  3 # 3 # 5
    num_gpus = 0.25
    num_cpus = 8
    fdate = "2022-06-15"

    @ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)  # This decorator indicates that this function will be distributed, with each task using one GPU.
    def train(config, jsize, sigma, N, lmbda, nu, S):
        # Here we randomly generate training data.
        X, Y, U = generate2D(jsize=jsize, sigma=sigma, N=N)

        if torch.cuda.is_available(): # cuda gpus
            device_id = torch.cuda.current_device() 
            device = torch.device("cuda:{}".format(device_id)) 
            torch.cuda.set_device(device)

        elif torch.backends.mps.is_available(): # mac gpus
            device = torch.device("mps")
            
        resolution = 1/int(np.sqrt(N))
        model = FDD(Y, X, level = S, lmbda = lmbda, nu = nu, iter = 5000, tol = 5e-5, resolution=resolution,
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

    
    dflist = []

    for sigma in [0.01, 0.05]: #, 0.05]:
        
        # calculate Cohen's d jump sizes
        X, Y, U = generate2D(jsize = 0, sigma=sigma, N=N_sure)
        std = np.std(Y)
        jsizes = np.array([0.25, 0.5, 0.75]) * std

        for jsize in jsizes: # , 0.2, 0.5]:

            print("Running SURE")
            # run SURE once for largest N
            X, Y, U = generate2D(jsize, sigma=sigma, N=N_sure)
            resolution = 1/int(np.sqrt(N_sure))
            model = FDD(Y, X, level = S, lmbda = 20, nu = 0.01, iter = 10000, tol = 5e-5, resolution=resolution, pick_nu = "MS", 
                        scaled = True)
            res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
                    num_gpus=num_gpus, num_cpus=num_cpus)
            best = res.get_best_result(metric = "score", mode = "min")

            config = best.metrics['config']
            lmbda, nu = config['lmbda'], config['nu']

            print("Running simulations")
            sims = list(range(num_sims))  # 100 simulations
            results = ray.get([train.remote(config, jsize, sigma, N, lmbda, nu, S) for config in sims for N in N_list])

            temp = pd.concat(results)
            dflist.append(temp)
            temp.to_csv("s3://ipsos-dvd/fdd/data/" + fdate + "/simulations_2d_sigma_" + str(sigma) + "_jsize_" + str(jsize) + ".csv", index=False)
            print(f"Done with sigma {sigma}, jump size {jsize}")
            
    # dflist = []
    
    # # get file names in s3 folder s3://ipsos-dvd/fdd/data/2022-06-09/    
    # s3 = boto3.resource('s3')
    # bucket = s3.Bucket('ipsos-dvd')
    # objs = bucket.objects.filter(Prefix="fdd/data/2022-06-09/")
    # files = [obj.key for obj in objs if obj.key.endswith(".csv")]
    
    # # loop over sigmas, jzies and files and run the simulations but not the SURE
    # for file in files:
    #     # load files
    #     df = pd.read_csv("s3://ipsos-dvd/" + file)
        
    #     # get parameters
    #     lmbda = df['lambda'].iloc[0]
    #     nu = df['nu'].iloc[0]
    #     jsize = df['alpha'].iloc[0]
    #     sigma = df['sigma'].iloc[0]
        
    #     print("Running simulations")
    #     sims = list(range(num_sims))  # 100 simulations
    #     results = ray.get([train.remote(config, jsize, sigma, N, lmbda, nu, S) for config in sims for N in N_list])
            
    #     temp = pd.concat(results)
    #     dflist.append(temp)
        
    #     # save to s3
    #     temp.to_csv("s3://ipsos-dvd/fdd/data/2022-06-14/simulations_2d_sigma_" + str(sigma) + "_jsize_" + str(jsize) + ".csv", index=False)
    #     print(f"Done with sigma {sigma}, jump size {jsize}")
            
    sys.stdout = old_stdout
    log_file.close()

    total = pd.concat(dflist)
    total.to_csv("s3://ipsos-dvd/fdd/data/" + fdate + "/simulations_2d.csv", index = False)
