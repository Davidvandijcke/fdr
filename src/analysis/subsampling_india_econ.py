import pandas as pd 
import os
import numpy as np
from matplotlib import pyplot as plt
from FDD import FDD
from FDD.SURE import SURE
import pickle
import boto3
import ray
from shapely.geometry import Polygon
import geopandas as gpd

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
    fn = '/Users/davidvandijcke/Dropbox (University of Michigan)/rdd/data/out/grid_subsampling/grid_subsampling.csv.gz'
    # s3.download_file('projects-fdd', 'data/out/' + fn, fn)
    gdf = pd.read_csv(fn)
    
    # def create_grid(min_x, max_x, min_y, max_y, step):
    #     grid = []
    #     for x in np.arange(min_x, max_x, step):
    #         for y in np.arange(min_y, max_y, step):
    #             grid.append(Polygon([(x, y), (x+step, y), (x+step, y+step), (x, y+step)]))
    #     return grid

    # # create the grid
    # min_x, min_y, max_x, max_y =[7649981.160830013, 2531547.754779711, 9015022.947024144, 3621930.062059432]
    # grid = create_grid(min_x, max_x, min_y, max_y, step=40000) # 5km grid

    # # create a GeoDataFrame from the grid
    # grid_gdf = gpd.GeoDataFrame(geometry=grid)
    # grid_gdf['x'] = grid_gdf.geometry.centroid.x
    # grid_gdf['y'] = grid_gdf.geometry.centroid.y
    
    # temp = gdf[['x', 'y']].drop_duplicates()
    # test = grid_gdf.merge(temp, on=['x', 'y'], how='left', indicator=True)
    # test = test[test['_merge'] == 'left_only'].drop(columns=['_merge'])
    # test = pd.concat([test.assign(post=c) for c in [0,1]], ignore_index=True)

    # # now add test to gdf
    # gdf = gdf.append(test)
    
    gdf['shop'] = gdf['shop'].fillna(0)
    
    gadm = gpd.read_file(os.path.join('/Users/davidvandijcke/Dropbox (University of Michigan)/rdd/data/in', 'india', "gadm41_IND_3.json"))
    gadm = gadm.to_crs('epsg:3857')
    
    # problem might be that apache sedona deduplicates geometries?

    df = gdf.copy()
    def aggregatePings(df):
        df_post = df[df['post'] == 1] 
        df_post_temp = df_post.groupby(['GID_3']).agg(pings=('pings_id', 'count'), shops=('shop', 'sum')).reset_index()
        # df_post_temp = df_post_temp.merge(gadm[['GID_2', 'geometry']], on='GID_2', how='left')
        # df_post_temp = gpd.GeoDataFrame(df_post_temp, geometry=df_post_temp.geometry)
        df_post_temp = df_post_temp.merge(gadm[['GID_3', 'geometry']], on='GID_3', how='left')
        df_post_temp = gpd.GeoDataFrame(df_post_temp, geometry=df_post_temp.geometry)
        df_post_temp['econ'] = df_post_temp['shops'] / df_post_temp['pings']
        df_post_temp.loc[df_post_temp.pings == 0, 'econ' ] = 0
        
        df_post = df_post.groupby(['x', 'y']).agg(shops_grid=("shop", "sum"), pings_grid =('pings_id', 'count')).reset_index()
        df_post = gpd.GeoDataFrame(df_post, geometry=gpd.points_from_xy(df_post.x, df_post.y), crs="EPSG:3857")
        df_post['geometry'] = df_post.buffer(20000, cap_style = 3)
        # # df_post = df_post.sjoin(gadm, how="left", op='intersects')
        
        # # df_post_temp['econ'] = df_post_temp['shops'] / df_post_temp['pings']
        # # df_post_temp.loc[df_post_temp.pings == 0, 'econ' ] = 0

        df_post = (df_post.sjoin(df_post_temp, how="left", op='intersects')
                        .groupby(['x', 'y']).agg(shops=('shops_grid', 'mean'), 
                                                 pings=('pings', 'mean'), 
                                                 econ=('econ', 'mean')).reset_index()
        )
        # df_post['econ'] = df_post['shops'] / df_post['pings']
        # df_post.loc[df_post.pings == 0, 'econ' ] = 0
        # df_post = df_post.groupby(['x', 'y']).agg(econ=('econ', 'mean')).reset_index()
        # df_post = df_post.drop(columns='econ').merge(df_post_temp, on='GID_2', how='left')
        
        # df_post = gpd.GeoDataFrame(df_post, geometry=gpd.points_from_xy(df_post.x, df_post.y), crs="EPSG:3857")
        
        # df_post = df_post.sjoin(df_post_temp, how="left", op='intersects').groupby(['x', 'y']).agg(shops=("shops", "mean"), 
        #                                                                                            pings=('pings', 'mean')).reset_index()
        
        df_pre = df[df['post'] == 0]
        # df_pre_temp = df_pre.groupby(['GID_2', 'date']).agg(pings=('pings_id', 'nunique'), shops=('shop', 'sum')).reset_index()
        # df_pre_temp['econ'] = df_pre_temp['shops'] / df_pre_temp['pings']
        # df_pre_temp.loc[df_pre_temp.pings == 0, 'econ' ] = 0
        # df_pre_temp = df_pre_temp.groupby(['GID_2']).agg(econ=('econ', 'mean')).reset_index()
        # df_pre_temp = df_pre_temp.merge(gadm[['GID_2', 'geometry']], on='GID_2', how='left')
        # df_pre_temp = gpd.GeoDataFrame(df_pre_temp, geometry=df_pre_temp.geometry)
        
        df_pre = df_pre.groupby(['x', 'y', 'date']).agg(shops=("shop", "sum"), pings=('pings_id', 'count')).reset_index()
        df_pre['econ'] = df_pre['shops'] / df_pre['pings']
        df_pre.loc[df_pre.pings == 0, 'econ' ] = 0
        df_pre = df_pre.groupby(['x', 'y']).agg(econ=('econ', 'mean'), shops=('shops', 'mean')).reset_index()
        # df_pre = gpd.GeoDataFrame(df_pre, geometry=gpd.points_from_xy(df_pre.x, df_pre.y), crs="EPSG:3857")
        # df_pre['geometry'] = df_pre.buffer(20000, cap_style = 3)
        # df_pre = df_pre.sjoin(gadm, how="left", op='intersects')
        


        # df_pre = df_pre.drop(columns="index_right").sjoin(df_pre_temp, how="left", op='intersects').groupby(['x', 'y']).agg(econ=('econ', 'mean')).reset_index()
        

        
        # df_pre['shops_unique'] = df_pre['shop'] * (1-df_pre['pings_id'].duplicated())
        # df_pre= df_pre.groupby(['x', 'y']).agg(shops=("shops_unique", "sum"), pings=('pings_id', 'nunique')).reset_index()
        # df_pre['econ'] = df_pre['shops'] / df_pre['pings']
        # df_pre.loc[df_pre.pings == 0, 'econ' ] = 0
        # df_pre = df_pre.groupby(['x', 'y']).agg(econ=('econ', 'mean')).reset_index()
        
        
        dfagg = df_pre.merge(df_post, on=['x', 'y'], suffixes=('_pre', '_post'))
        dfagg['pings_share_change'] = dfagg['econ_post'] - dfagg['econ_pre']
        dfagg['pings_share_norm'] = dfagg['pings_share_change'] / dfagg['econ_pre']
        dfagg['pings_share_ratio'] = dfagg['econ_post'] / dfagg['econ_pre']

        dfagg.loc[dfagg.econ_pre == 0, 'pings_share_norm' ] = 0
        dfagg.loc[dfagg.econ_pre == 0, 'pings_share_ratio' ] = 0
        dfagg['shop_ratio'] = 

        return df
    
    import geopandas as gpd
    temp = gpd.GeoDataFrame(dfagg, geometry=gpd.points_from_xy(dfagg.x, dfagg.y))
    temp['geometry'] = temp.buffer(20000, cap_style = 3)
    
    
    states_gdf = gpd.read_file(os.path.join('/Users/davidvandijcke/Dropbox (University of Michigan)/rdd/data/in', 'india', 'india_states_shapefile'))

    # Convert the CRS of the states GeoDataFrame to match the CRS of the Rajasthan GeoDataFrame
    states_gdf = states_gdf.to_crs('epsg:3857')
    

    # Extract the geometry for Rajasthan
    rajasthan_geometry = states_gdf[states_gdf['name_1'] == 'Rajasthan']['geometry']

   # Reset the limits for the zoom (to show the entire Rajasthan)
    rajasthan_extent = states_gdf[states_gdf['name_1'] == 'Rajasthan'].total_bounds
    xlim = (rajasthan_extent[0], rajasthan_extent[2])
    ylim = (rajasthan_extent[1], rajasthan_extent[3])

    # Create the plot
    fig, ax = plt.subplots(1, 1)

    # Plot the data
    vmax = 1.5
    p = temp.plot(column="pings_share_norm", cmap='coolwarm_r',  ax=ax, legend=True, vmax=2)

    # Plot the Rajasthan border
    rajasthan_geometry.boundary.plot(color='k', linewidth=2, ax=ax, alpha=0.5)

    

    temp.plot(column="pings_share_norm", legend=True, cmap="coolwarm_r", vmax=0)
    temp['test'] = temp['econ_pre'] == 0
    temp.plot(column="test", legend=True, cmap="coolwarm_r")


    
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
    
        

