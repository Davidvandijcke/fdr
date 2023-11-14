import pandas as pd 
import os
import numpy as np
from matplotlib import pyplot as plt
from FDD import FDD
from FDD.SURE import SURE
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
    
    s3 = boto3.client('s3')


    
    # get SURE parameters
    fn = 'india_econ_SURE_90_lambda25_nu025.pkl'
    fto = os.path.join(data_out, fn) 
    
    with open(fn, 'wb') as f:
        s3.download_fileobj('projects-fdd', 'data/out/' + fn, f)
    with open(fn, "rb") as f:
        res = pickle.load(f)
        
    best = res.get_best_result(metric = "score", mode = "min")

    config = best.metrics['config']
    lmbda, nu = config['lmbda'], config['nu']

    ## get raw data for subsampling
    # pull it down from s3 first
    bucket = 'projects-fdd'
    fn = 'grid_subsampling.csv.gz'
    s3.download_file('projects-fdd', 'data/out/' + fn, fn)
    gdf = pd.read_csv(fn)
    gdf['post'] = (gdf['date'] == "2021-09-26").astype(int)
    
    df = gdf.copy()
    def aggregatePings(df):
        df_pre = df[df['post'] == 0]
        df_post = df[df['post'] == 1]
        df_pre = df_pre.groupby(['x', 'y']).agg(econ=("shop", "mean")).reset_index()
        df_post = df_post.groupby(['x', 'y']).agg(econ=("shop", "mean")).reset_index()
        df = df_pre.merge(df_post, on=['x', 'y'], suffixes=('_pre', '_post'))
        df['pings_share_change'] = df['econ_post'] - df['econ_pre']
        df['pings_share_norm'] = df['pings_share_change'] / df['econ_pre']
        df.loc[df.econ_pre == 0, 'pings_share_norm' ] = 0
        
        return df
    
    import geopandas as gpd
    temp = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y))

    
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
    
        

