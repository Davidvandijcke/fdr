from FDR import FDR
from FDR.SURE import SURE
import geopandas as gpd
import numpy as np
import os
import pickle

# set path
bsdir = "/home/dvdijcke/"
data_out =  bsdir + "data/out/ukr/" # "s3://ipsos-dvd/fdd/data/out/ukr/" # set to desired directory
data_in = bsdir + "data/in/ukr/"

def getSpaceTimeDwells(start, end):
    """Get the spatio-temporal data for the frontline dwells
    Args:
        start (int): first week of 2022 to consider
        end (int): last week of 2022 to consider
    """
    grid = gpd.read_file(data_out + "grid_dwells_weekly.geojson")
    grid = grid.set_crs("EPSG:6381", allow_override=True)
    
    # assign first-period matrices
    no_weeks = len(range(start,end))
    agg_gdf = grid[grid.week==start]
    X1 = np.array(agg_gdf.geometry.centroid.x)
    X2 = np.array(agg_gdf.geometry.centroid.y)
    X = np.stack([X2, X1]).T 
    Y = np.array(agg_gdf['count_ratio'])

    # assign spatio-temporal matrices
    Y_t = Y.copy()
    
    # Step 1: Duplicate the array N times
    duplicated_array = np.tile(X, (no_weeks, 1))

    # Step 2: Create a column indicating the repetition number
    repetition_column = np.repeat(np.arange(start, end), len(X)).reshape(-1, 1)

    # Step 3: Concatenate the duplicated array with the repetition column
    X_t = np.hstack((duplicated_array, repetition_column))
    
    
    for week in range(start+1,end):
        agg_gdf = grid[grid.week==week]
        Y = np.array(agg_gdf['count_ratio'])
        Y_t = np.hstack([Y_t, Y]) # first week is first, then chronologically stacked
        
    # winsorize
    qtile = np.quantile(Y_t, 0.95)
    Y_t[Y_t > qtile] = qtile # 0.5
    
    return(X, X_t, Y_t, no_weeks)

if __name__ == "__main__":

    #----------------
    # parameters
    #----------------
    sigma=0.05
    S = 32
    N = 1000
    lmbda = 50
    nu = 0.05
    num_samples = 2 # 225 #  400 # 400 # 400 # 200
    R = 1 # 3 # 3 # 3 # 5
    num_gpus = 1
    num_cpus = 2
    iter = 50000
    lmbda_max = 50
    nu_min = 0.001
    nu_max = 1
    
    #----------------
    # process data
    #----------------
    
    start = 9
    end = 52
    X, X_t, Y_t, no_weeks = getSpaceTimeDwells(start, end)

    # segment
    #--------

    #--------
    grid_n = np.array([no_weeks, len(np.unique(X[:,0])), len(np.unique(X[:,1]))]).T
    model = FDR(Y_t, X_t, level = S, lmbda = lmbda, nu = nu, iter = iter, tol = 5e-5, pick_nu = "MS", 
                CI=False, rectangle=True, grid_n=grid_n, scripted=False)

    test = model.run()
    file_name = 'frontline_dwells_temporal_SURE.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(test, file)
    
    # res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
    #     num_gpus=num_gpus, num_cpus=num_cpus, lmbda_max=lmbda_max,
    #     nu_min=nu_min, nu_max=nu_max)

    # file_name = 'frontline_dwells_temporal_SURE.pkl'
    # with open(file_name, 'wb') as file:
    #     pickle.dump(res, file)