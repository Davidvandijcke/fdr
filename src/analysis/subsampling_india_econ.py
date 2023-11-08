import pandas as pd 
import os
import numpy as np
from matplotlib import pyplot as plt
from FDD import FDD
from FDD.SURE import SURE
import geopandas as gpd
import pickle
import boto3
import ray

def moveUp(fn, times = 1):
    for _ in range(times):
        fn = os.path.dirname(fn)
    return fn

if __name__ == '__main__': 
    
     #----------------
    # parameters
    #----------------
    S = 32
    N = 1000
    num_gpus = 1
    num_cpus = 2
    
    # get directory above
    main_dir = "s3://projects-fdd/" # moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    # get SURE parameters
    fn = 'india_econ_SURE_90_lambda25_nu025.pkl'
    fto = os.path.join(data_out, fn) 
    s3 = boto3.client('s3')
    
    with open(fn, 'wb') as f:
        s3.download_fileobj('projects-fdd', 'data/out/' + fn, f)
    with open(fn, "rb") as f:
        res = pickle.load(f)
        
    best = res.get_best_result(metric = "score", mode = "min")

    config = best.metrics['config']
    lmbda, nu = config['lmbda'], config['nu']

    fn_merged = os.path.join(data_out, 'india', 'rajasatan_cheating_shops_merged_40K.geojson')
    gdf = gpd.read_file(fn_merged)
    
    gdf['pings_norm'] = gdf['pings_norm'] * 100
    print(gdf.head())
    Y = np.array(gdf['pings_norm'])
    X = np.stack([np.array(gdf.geometry.centroid.x), np.array(gdf.geometry.centroid.y)]).T

    qtile = np.quantile(Y, 0.90)
    Y[Y>qtile] = qtile
        
    resolution = 1/int(np.sqrt(Y.size))
    model = FDD(Y, X, level = 32, lmbda = lmbda, nu = nu, iter = 10000, tol = 5e-5, resolution=resolution,
        pick_nu = "MS", scaled = True, scripted = False, rectangle=True)
    # results = model.run()
    # u = results['u']
    # fn = "india_u.npy"
    # np.save(fn, u)
    
    
    # s3 = boto3.client('s3')
    # with open(fn, "rb") as f:
    #     s3.upload_fileobj(f, "projects-fdd", "data/out/subsampling/" + fn)
    
    # ray.init()
    (test, b) = model.subSampling(nboot = 2)    
    test = np.stack(test, axis=0)
    fn = "india_boots.npy"
    np.save(fn, test)
    with open(fn, "rb") as f:
        s3.upload_fileobj(f, "projects-fdd", "data/out/subsampling/" + fn)
        
    fn = "india_b.npy"
    np.save(fn, b)
    with open(fn, "rb") as f:
        s3.upload_fileobj(f, "projects-fdd", "data/out/subsampling/" + fn)
    
        

