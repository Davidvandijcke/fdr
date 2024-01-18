import pandas as pd 
import os
import numpy as np
from matplotlib import pyplot as plt
from rasterio.mask import raster_geometry_mask, mask
import cv2
from FDD import FDD
from FDD.SURE import SURE
import geopandas as gpd
import pickle
import boto3


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
    lmbda = 1000
    nu = 0.02
    num_samples = 400 # 225 #  400 # 400 # 400 # 200
    R =  3 # 3 # 3 # 3 # 5
    num_gpus = 1
    num_cpus = 2


    # get directory above
    main_dir = "/home/dvdijcke/" # moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    fn_merged = os.path.join(data_out, 'india', 'rajasatan_cheating_shops_merged_40K_devices_agg.csv')
    gdf = pd.read_csv(fn_merged)
    # gdf = gdf.to_crs('epsg:3857')

    gdf['shops_ratio'] = gdf['shops_ratio'] * 100
    # gdf['shops_ratio'] = np.where(gdf['count_before'] == 0, 100, gdf['count_norm'])
    print(gdf.head())
    Y = np.array(gdf['shops_ratio'])
    X = np.stack([np.array(gdf.y), np.array(gdf.x)]).T

    qtile = 150 # np.quantile(Y, 0.85)
    Y[Y>qtile] = qtile
        
    resolution = 1/int(np.sqrt(Y.size))
    model = FDD(Y, X, level = 32, lmbda = 150, nu = 0.008, iter = 10000, tol = 5e-5, resolution=resolution,
        pick_nu = "MS", scaled = True, scripted = False, rectangle=True)
    res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
        num_gpus=num_gpus, num_cpus=num_cpus, lmbda_max=5, nu_min=0.05, nu_max=0.1) # lmbda = 5, nu_min = 0.07 

    file_name = os.path.join(data_out, 'india_econ_SURE_devices_lambda5_nu05-01.pkl')
    with open(file_name, 'wb') as file:
        pickle.dump(res, file)
        
