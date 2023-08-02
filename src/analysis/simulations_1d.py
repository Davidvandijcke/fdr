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

def ft(x, jumps):
    temp = x**2 + np.sin(10 * x)  # change 10 to any other value to adjust the frequency of oscillation
    for (xj, jj) in jumps:
        if x > xj:
            temp += jj
    return temp

# Redefine the function to generate data with larger jumps, with last one going down
def generate1D(jumps=[(0.2013934, 0.6), (0.4023231, 1), (0.590349, 1.5), (0.7893434, -2)], sigma=0.1, N=500):
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
        temp_bdy += ((data+model.resolution) < jump) * ((np.append(data[1:], 0) + model.resolution) > jump)

    x, y, u = generate1D(N = data.size)


    return u, temp_bdy



if __name__ == "__main__":

    #----------------
    # parameters
    #----------------
    N_list = [500, 1000, 5000]
    N_sure = max(N_list)
    S = 32
    num_samples = 400 #  400 # 400 # 200
    num_sims = 100 # 100 # 100 # 100
    R = 3 #  3 # 3 # 5
    num_gpus = 1
    num_cpus = 4
    fdate = "2022-08-02"
    main_dir = "/home/dvdijcke/"
    data_out = os.path.join(main_dir, "data", "out")
    SURE = False

    @ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)  # This decorator indicates that this function will be distributed, with each task using one GPU.
    def train(config, sigma, N, lmbda, nu, S):
        # Here we randomly generate training data.
        X, Y, U = generate1D(sigma=sigma, N=N)
        
        #tune.utils.wait_for_gpu(target_util = 0.1, retry = 100000)


        if torch.cuda.is_available(): # cuda gpus
            device_id = torch.cuda.current_device() 
            device = torch.device("cuda:{}".format(device_id)) 
            torch.cuda.set_device(device)

        elif torch.backends.mps.is_available(): # mac gpus
            device = torch.device("mps")
            
        resolution = 1/int(N*0.05)
        model = FDD(Y, X, level = S, lmbda = lmbda, nu = nu, iter = 100000, tol = 5e-5, resolution=resolution,
                pick_nu = "MS", scaled = True, scripted = False)
        
        u, jumps, J_grid, nrj, eps, it = model.run()
        
        u_original, J_original = getOriginalImage(model)

        
        mse = np.mean((u - u_original)**2)
        jump_pos = np.sum(J_grid * (1-J_original)) / np.sum(1-J_original) # false positive rate (significance)
        jump_neg = np.sum((1-J_grid) * (J_original)) / np.sum(J_original) # false negative rate (1-power)
        
        temp = pd.DataFrame(jumps)
        temp[['N', 'S', 's', 'sigma', 'lambda', 'nu', 'jump_neg', 
              'jump_pos', 'mse']] = N, S, config, sigma, lmbda, nu, jump_neg, jump_pos, mse
        return temp

    np.random.seed(50345)

    dflist = []

    for sigma in [0.05]: #, 0.05]:

        print("Running SURE")
        # run SURE once for largest N
        X, Y, U = generate1D(sigma=sigma, N=N_sure)
        resolution = 1/int(0.05*N_sure)
        model = FDD(Y, X, level = S, lmbda = 20, nu = 0.01, iter = 10000, tol = 5e-5, pick_nu = "MS", 
                    scaled = True, resolution=resolution, scripted=False)
        
        if SURE:
            res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
                    num_gpus=num_gpus, num_cpus=num_cpus, nu_min = 0.0001, nu_max = 0.01)
            best = res.get_best_result(metric = "score", mode = "min")

            config = best.metrics['config']
            lmbda, nu = config['lmbda'], config['nu']
        
            del(model)
            torch.cuda.empty_cache()
        else:
            df = pd.read_csv(os.path.join(data_out, "simulations", "2022-07-31", "simulations_1d_sigma_0.05.csv"))
            lmbda, nu, sigma, S = df[['lambda', 'nu', 'sigma', 'S']].loc[0]

        # lmbda = 120
        # nu = 0.0016
        # model.lmbda = lmbda
        # model.nu = nu
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
        results = ray.get([train.remote(config, sigma, N, lmbda, nu, S) for config in sims for N in N_list])

        temp = pd.concat(results)
        dflist.append(temp)
        temp.to_csv(os.path.join(data_out, "simulations/" + fdate + "/simulations_1d_sigma_" + str(sigma) + ".csv"), index=False)
        
        print(f"Done with sigma {sigma}")