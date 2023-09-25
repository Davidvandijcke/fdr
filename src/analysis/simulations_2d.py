from FDD import FDD
from FDD.SURE import SURE
import numpy as np
import pandas as pd
import torch 
from functools import partial
from matplotlib import pyplot as plt
import ray
from ray import tune
import boto3
import pickle
import os
from ray.air.config import RunConfig
from ray.tune import CLIReporter, JupyterNotebookReporter



def f(x,y, jsize):
  temp = np.sqrt((x-1/2)**2 + (y-1/2)**2)
  if temp < 1/4:
      return temp
  else:
      return temp + jsize
  
def get_reporter(max_progress_rows=10, metric_column="custom_metric"):
    # Check if ipykernel is loaded
    is_jupyter = 'ipykernel' in sys.modules

    if is_jupyter:
        reporter = JupyterNotebookReporter(overwrite=True)
    else:
        reporter = CLIReporter(max_progress_rows=max_progress_rows)

    reporter.add_metric_column(metric_column)

    return reporter

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

    # log_file = open("message.log","w")

    # sys.stdout = log_file

    #-------------
    # parameters
    #-------------
    N_list = [5000, 10000, 20000]
    N_sure = max(N_list)
    S = 32
    num_samples = 400 # 400 # 200
    num_sims = 100 # 100 # 100
    R = 1 #  3 # 3 # 5
    num_gpus = 0.5
    num_cpus = 3
    fdate = "2023-09-24"

    @ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)  # This decorator indicates that this function will be distributed, with each task using one GPU.
    def train_func(config, jsize, sigma, lmbda, nu, S):
        s = config['s']
        N = config['N']
        print(f"s = {s}, N = {N}")
        # Here we randomly generate training data.
        X, Y, U = generate2D(jsize=jsize, sigma=sigma, N=N)
        
        #tune.utils.wait_for_gpu(target_util = 0.1, retry = 100000)


        if torch.cuda.is_available(): # cuda gpus
            device_id = torch.cuda.current_device() 
            device = torch.device("cuda:{}".format(device_id)) 
            torch.cuda.set_device(device)

        elif torch.backends.mps.is_available(): # mac gpus
            device = torch.device("mps")
            
        resolution = 1/int(np.sqrt(N*0.05))
        model = FDD(Y, X, level = S, lmbda = lmbda, nu = nu, iter = 15000, tol = 5e-5, resolution=resolution,
                pick_nu = "MS", scaled = True, scripted = False, CI=False)
        
        results = model.run()
        u = results['u']
        J_grid = results['J']
        jumps = results['jumps']
        it = results['it']
        
        print(f"Iterations {it}")
        
        u_original, J_original = getOriginalImage(model, jsize)

        
        mse = np.mean((u - u_original)**2)
        jump_pos_true = (J_original + np.vstack([J_original[1:, :], np.zeros((1, J_original.shape[1]))]) + np.hstack([J_original[:, 1:], np.zeros((J_original.shape[0],1))]))
        jump_pos = np.sum(J_grid * (1-(jump_pos_true>0))) / np.sum(1-J_original) # false positive rate (significance)
        jump_neg_true = (J_grid + np.vstack([J_grid[1:, :], np.zeros((1, J_grid.shape[1]))]) + np.hstack([J_grid[:, 1:], np.zeros((J_grid.shape[0],1))]))
        jump_neg = np.sum((1-(jump_neg_true>0)) * (J_original)) / np.sum(J_original) # false negative rate (1-power)
                
        temp = pd.DataFrame(jumps)
        temp[['alpha', 'N', 'S', 's', 'sigma', 'lambda', 'nu', 'jump_neg', 
              'jump_pos', 'mse']] = jsize, N, S, s, sigma, lmbda, nu, jump_neg, jump_pos, mse
        torch.cuda.empty_cache()
        
        #temp.to_csv("s3://projects-fdd/data/out/simulations/" + fdate + "/simulation_2d_N" + str(N) + "_sim" + str(config) + "_" + str(sigma) + "_jsize_" + str(jsize) + ".csv", index=False)        
        return temp

    
    dflist = []

    for sigma in [0.05]: #, 0.05]:
        
        # calculate Cohen's d jump sizes
        X, Y, U = generate2D(jsize = 0, sigma=sigma, N=N_sure)
        std = np.std(Y)
        jsizes = np.array([0.25, 0.5, 0.75]) * std

        for jsize in jsizes: # , 0.2, 0.5]:

            print("Running SURE")
            # run SURE once for largest N
            X, Y, U = generate2D(jsize, sigma=sigma, N=N_sure)
            resolution = 1/int(np.sqrt(0.05*N_sure))
            model = FDD(Y, X, level = S, lmbda = 20, nu = 0.01, iter = 15000, tol = 5e-5, pick_nu = "MS", 
                        scaled = True, resolution=resolution, scripted=False, CI=False)
            res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
                    num_gpus=num_gpus, num_cpus=num_cpus, nu_max=0.1, nu_min = 0.001)
            
            # file_name = 'jsize' + str(jsize) + '_N' + str(N_sure) + '.pkl'
            # with open(file_name, 'wb') as file:
            #     pickle.dump(res, file)
                
            # s3 = boto3.client('s3')
            # with open(file_name, "rb") as f:
            #     s3.upload_fileobj(f, "projects-fdd", "data/out/" + file_name)
            best = res.get_best_result(metric = "score", mode = "min")
            torch.cuda.empty_cache()

            config = best.metrics['config']
            lmbda, nu = config['lmbda'], config['nu']
            
            model.lmbda = lmbda
            model.nu = nu
            # model.tol = 5e-6
            # u, jumps, J_grid, nrj, eps, it = model.run()
            # temp = pd.DataFrame(jumps)
            # temp['Y_jumpsize'].abs().mean()
            
            # plt.hist(temp['Y_jumpsize'])
            # plt.show()
            
            # test = temp[temp['Y_jumpsize'].abs() < 0.02]
            # plt.scatter(test['X_0'], test['X_1'], color = "blue")
            # test = temp[temp['Y_jumpsize'].abs() > 0.02]
            # plt.scatter(test['X_0'], test['X_1'], color = "red")
            print("Running simulations")
            sims = list(range(num_sims))  # 100 simulations
            results = ray.get([train_func.remote({'s' : s, "N" : N}, jsize, sigma, lmbda, nu, S) for s in sims for N in N_list])

            temp = pd.concat(results)
            dflist.append(temp)
            temp.to_csv("s3://projects-fdd/data/out/simulations/" + fdate + "/simulation_2d_" + str(sigma) + "_jsize_" + str(jsize) + ".csv", index=False)        
            
            print(f"Done with sigma {sigma}, jump size {jsize}")
            
    dflist = []
    
    # get file names in s3 folder s3://ipsos-dvd/fdd/data/2022-06-09/    
    # s3 = boto3.resource('s3')
    # bucket = s3.Bucket('ipsos-dvd')
    # objs = bucket.objects.filter(Prefix="fdd/data/2022-06-09/")
    
    # s3_resource = boto3.resource('s3')

    # # List objects
    # bucket = "projects-fdd"
    # files = s3_resource.Bucket(bucket).objects.filter(Prefix='data/out/simulations/2022-08-02/')
    
    # # loop over sigmas, jzies and files and run the simulations but not the SURE
    # for file in files:
    #     if "csv" in file.key:

    
    #         # load files
    #         df = pd.read_csv(os.path.join("s3://", bucket, file.key))
            
    #         # get parameters
    #         lmbda = df['lambda'].iloc[0]
    #         nu = df['nu'].iloc[0]
    #         jsize = df['alpha'].iloc[0]
    #         sigma = df['sigma'].iloc[0]
            
    #         print("Running simulations")
    #         sims = list(range(num_sims))  # 100 simulations
            
    #         search_space={
    #             # A random function
    #             "N": tune.grid_search(N_list),
    #             "s":  tune.grid_search(sims)
    #             # Use the `spec.config` namespace to access other hyperparameters
    #             #"nu":
    #         }
    #         trainable_with_resources = tune.with_resources(
    #             partial(train_func, jsize=jsize, sigma=sigma, lmbda=lmbda, nu=nu, S=S), 
    #             {"cpu": num_cpus, "gpu": num_gpus}
    #         )
    #                 # Start the Ray Tune run
    #         analysis = tune.Tuner(
    #             trainable_with_resources,
    #             param_space=search_space,
    #             run_config=RunConfig(progress_reporter=get_reporter())
    #         )

    #         # Get the hyperparameters of the best trial
    #         results = analysis.fit() #get_best_trial("objective", "min", "last")
                
    #         # temp = pd.concat(results)
    #         # dflist.append(temp)
            
    #         # # save to s3
    #         # temp.to_csv("/home/dvdijcke/data/out/simulations/2022-07-07/" + str(sigma) + "_jsize_" + str(jsize) + ".csv", index=False)
    #         print(f"Done with sigma {sigma}, jump size {jsize}")
            
    # # sys.stdout = old_stdout
    # # log_file.close()

    # total = pd.concat(dflist)
    #total.to_csv("s3://ipsos-dvd/fdd/data/" + fdate + "/simulations_2d.csv", index = False)
