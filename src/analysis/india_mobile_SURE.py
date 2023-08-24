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
    num_samples = 2 # 225 #  400 # 400 # 400 # 200
    R =  1 # 3 # 3 # 3 # 5
    num_gpus = 0.5
    num_cpus = 2


    # get directory above
    main_dir = "s3://ipsos-dvd/fdd" # "/home/dvdijcke/" # moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    fn_merged = os.path.join(data_out, 'india', 'rajasatan_cheating_grid.geojson')
    gdf = gpd.read_file(fn_merged)
    
    #gdf = gdf.to_crs("epsg:3857")
    gdf = gdf[~gdf.geometry.isna()]
    gdf = gdf[~gdf.geometry.isnull()]

    
    gdf['count_norm'] = gdf['count_norm'] * 100
    Y = np.array(gdf['count_norm'])
    X = np.stack([np.array(gdf.geometry.centroid.x), np.array(gdf.geometry.centroid.y)]).T
    
    qtile = np.quantile(Y, 0.9)
    Y[Y>qtile] = qtile
        
    resolution = 1/int(np.sqrt(0.25*Y.size))
    model = FDD(Y, X, level = 32, lmbda = 150, nu = 0.008, iter = 10000, tol = 5e-5, resolution=resolution,
        pick_nu = "MS", scaled = True, scripted = False, rectangle=True)
    res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
        num_gpus=num_gpus, num_cpus=num_cpus, lmbda_max=10) # can't remember if I set nu to 0.001, or 0.002

    file_name = 'india_mobile_SURE_95_025_lambda100.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(res, file)
        
    s3 = boto3.client('s3')
    with open(file_name, "rb") as f:
        s3.upload_fileobj(f, "ipsos-dvd", "fdd/data/out/" + file_name)