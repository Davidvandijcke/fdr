import pandas as pd 
import os
import numpy as np
from matplotlib import pyplot as plt
# from rasterio.mask import raster_geometry_mask, mask
# import cv2
from FDD import FDD
from FDD.SURE import SURE
import geopandas as gpd
import pickle
import boto3
import ray
import random



# def subSampling(nboot=300, Y, X)
#     boots = list(range(nboot))
#     n = Y.shape[0]
#     N = self.grid_x.size
#     b = sorted(np.random.randint(low=2*N, high=2*N+0.1*n, size=4))
#     I = list(range(self.Y_raw.shape[0]))
#     bootstrap_trial_dynamic = self.bootstrap_trial_factory(num_gpus=self.num_gpus, num_cpus=self.num_cpus)
#     results = ray.get([bootstrap_trial_dynamic.remote(self, b, I, s) for s in boots])



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
    nboot = 500


    # get directory above
    main_dir = "s3://projects-fdd/" #  "/home/dvdijcke/" # moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    fn = os.path.join(data_out, 'india', 'rajasatan_cheating_shops_merged_40K_devices (1).csv')
    # download from s3
    
    devices = pd.read_csv(fn)
    
    # first get parameters of original problem
    fn_merged = os.path.join(data_out, 'india', 'rajasatan_cheating_shops_merged_40K_devices_agg.csv')
    gdf = pd.read_csv(fn_merged)
    # gdf = gdf.to_crs('epsg:3857')

    gdf['shops_ratio'] = gdf['shops_ratio'] * 100
    # gdf['shops_ratio'] = np.where(gdf['count_before'] == 0, 100, gdf['count_norm'])
    Y = np.array(gdf['shops_ratio'])
    X = np.stack([np.array(gdf.y), np.array(gdf.x)]).T

    qtile = 150 # np.quantile(Y, 0.85)
    Y[Y>qtile] = qtile
    
    # get SURE parameters
    s3 = boto3.client('s3')
    fto = "parms.pkl"
    s3.download_file("projects-fdd", "data/out/india/india_econ_SURE_devices_lambda5_nu075-015.pkl", fto)
    with open(fto, "rb") as f:
        res = pickle.load(f)
    # best = res.get_best_result(metric = "score", mode = "min")

    # config = best.metrics['config']
    # lmbda, nu = config['lmbda'], config['nu']
    
    lmbda, nu = 4.98126140267338, 0.07543764833087317 # setting it manually because of ray version conflict with get_best_result function and don't want to restart everything
    
        
    resolution = 1/int(np.sqrt(Y.size))
    model = FDD(Y, X, level = 32, lmbda = lmbda, nu = nu, iter = 10000, tol = 5e-5, 
                resolution=resolution, pick_nu = "MS", scaled = True, 
                scripted = False, rectangle=True, CI=False)
    
    boots = list(range(nboot))
    n = devices.shape[0]
    b = sorted(np.random.randint(low=0.5*n, high=0.6*n, size=4))
    I = list(range(n))
    
    
    crs = "epsg:3857"

    gadm = gpd.read_file(data_out + "/india/gadm41_IND_3.json")
    gadm = gadm.to_crs(crs)

    def aggregateRawPings(df, gadm):
        gidname = "GID_3"
        df_post = df[df['post'] == 1] 

        #-- 1
        df_post_temp = df_post.groupby([gidname]).agg(shops=("shop", "sum")).reset_index()
        df_post_temp = df_post_temp.merge(gadm[[gidname, 'geometry']], on=gidname, how='left')
        df_post_temp = gpd.GeoDataFrame(df_post_temp, geometry=df_post_temp.geometry, crs=crs)
        df_post = df_post.groupby(['x', 'y']).agg(shops_grid=("shop", "sum")).reset_index()
        df_post = gpd.GeoDataFrame(df_post, geometry=gpd.points_from_xy(df_post.x, df_post.y), crs="EPSG:3857")
        df_post['geometry'] = df_post.buffer(20000, cap_style = 3)
        df_post = (df_post.sjoin(df_post_temp, how="left", op='intersects')
                        .groupby(['x', 'y']).agg(shops_grid=('shops_grid', 'mean'), 
                                                shops=('shops', 'mean')).reset_index()
        )
        df_pre = df[df['post'] == 0]

        # --- 1 
        df_pre_temp = df_pre.groupby([gidname, 'caid', 'date']).agg(shops=("shop", "sum")).reset_index()
        df_pre_temp = df_pre_temp.groupby([gidname, 'caid']).agg(shops=("shops", "median")).reset_index()
        # df_pre_temp = df_pre.groupby([gidname, 'date']).agg(shops=("shop", "sum")).reset_index()
        df_pre_temp = df_pre_temp.groupby([gidname]).agg(shops=("shops", "sum")).reset_index()


        df_pre_temp = df_pre_temp.merge(gadm[[gidname, 'geometry']], on=gidname, how='left')
        df_pre_temp = gpd.GeoDataFrame(df_pre_temp, geometry=df_pre_temp.geometry, crs=crs)
        df_pre = df_pre.groupby(['x', 'y']).agg(shops_grid=("shop", "sum")).reset_index()
        df_pre = gpd.GeoDataFrame(df_pre, geometry=gpd.points_from_xy(df_pre.x, df_pre.y), crs="EPSG:3857")
        df_pre['geometry'] = df_pre.buffer(20000, cap_style = 3)
        df_pre = (df_pre.sjoin(df_pre_temp, how="left", op='intersects')
                        .groupby(['x', 'y']).agg(shops_grid=('shops_grid', 'mean'), 
                                                shops=('shops', 'mean')).reset_index()
        )

        dfagg = df_pre.merge(df_post, on=['x', 'y'], suffixes=('_pre', '_post'))
        dfagg['shops_ratio'] = dfagg['shops_post'] / dfagg['shops_pre']
        # dfagg['shops_change'] = dfagg['shops_post'] / dfagg['shops_pre'] - 1
        dfagg.loc[dfagg.shops_ratio == np.inf, 'shops_ratio' ] = 1
        dfagg.loc[dfagg.shops_ratio.isna(), 'shops_ratio' ] = 1
        
        return dfagg
    


    def bootstrap_trial_factory(num_gpus, num_cpus):
        @ray.remote(num_gpus=num_gpus, num_cpus=num_cpus)
        def bootstrap_trial(devices, model, b, I, s):
            res = np.zeros((len(b)*len(s),) + model.grid_y.squeeze().shape)
            for i in s:
                I_star = I.copy()
                for j in range(len(b)-1, -1, -1):
                    I_star = random.sample(I_star, b[j])
                    
                    devices_sample = devices.loc[I_star].copy()

                    
                    dfagg = aggregateRawPings(devices_sample, gadm)
                    
                    dfagg['shops_ratio'] = dfagg['shops_ratio'] * 100
                    Y_star = np.array(dfagg['shops_ratio'])
                    X_star = np.stack([np.array(dfagg.y), np.array(dfagg.x)]).T

                    qtile = 150 # np.quantile(Y, 0.85)
                    Y_star[Y_star>qtile] = qtile

                    print(f"Running trial {i}, sample {b}")
                    model_temp = FDD(Y_star, X_star, level = model.level, lmbda = model.lmbda, nu = model.nu, iter = model.iter, tol = model.tol, resolution=model.resolution, pick_nu = "MS", scaled = True, scripted = False, rectangle=True, CI=False)
                    results = model_temp.run()
                    print(f"Done with trial {i}, sample {b}")
                    res[j+(i*len(b)),...] = results['u'] 
            return res
        return bootstrap_trial

    
    # run the original model once just to have the original function estimate
    results = model.run()
    u = results['u']
    fn = "india_econ_u.npy"
    np.save(fn, u)
    
    with open(fn, "rb") as f:
        s3.upload_fileobj(f, "projects-fdd", "data/out/subsampling/" + fn)

    ncores = 8 # want to pass data to cores only once, so set this to number of gpus
    s_chunks = [range(int(np.ceil(nboot/ncores))) for i in range(ncores)]
       
    # run the subsampling 
    bootstrap_trial_dynamic = bootstrap_trial_factory(num_gpus=num_gpus, num_cpus=num_cpus)
    test = ray.get([bootstrap_trial_dynamic.remote(devices, model, b, I, s) for s in s_chunks]) # boots
    
    test = np.stack(test, axis=0)
    fn = "india_econ_boots.npy"
    np.save(fn, test)
    with open(fn, "rb") as f:
        s3.upload_fileobj(f, "projects-fdd", "data/out/subsampling/" + fn)
        
    fn = "india_econ_boots_b.npy"
    np.save(fn, b)
    with open(fn, "rb") as f:
        s3.upload_fileobj(f, "projects-fdd", "data/out/subsampling/" + fn)