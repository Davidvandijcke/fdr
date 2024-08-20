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
    S = 16
    N = 1000
    lmbda = 50
    nu = 0.05
    num_samples = 100 # 225 #  400 # 400 # 400 # 200
    R = 1 # 3 # 3 # 3 # 5
    num_gpus = 1
    num_cpus = 4
    iter = 100000
    lmbda_max = 50
    nu_min = 0.001
    nu_max = 1
    
    #----------------
    # process data
    #----------------
    
    start = 9
    end = 51
    X, X_t, Y_t, no_weeks = getSpaceTimeDwells(start, end)

    # segment
    #--------
    
    # import SURE parameters
    file = open('frontline_dwells_temporal_SURE_9_51.pkl','rb')
    res = pickle.load(file)
    file.close()
    best = res.get_best_result(metric = "score", mode = "min")
    config = best.metrics['config']
    lmbda, nu = config['lmbda'], config['nu']

    print(f"lambda {lmbda}, nu {nu}")


    #--------
    grid_n = np.array([len(np.unique(X[:,0])), len(np.unique(X[:,1])), no_weeks])
    model = FDR(Y_t, X_t, level = S, lmbda = lmbda, nu = nu, iter = iter, tol = 5e-5, pick_nu = "MS", 
                CI=False, rectangle=False, grid_n=grid_n, scripted=False) 
    
    # TODO: conformal pred is not working right now
    
    # File "/home/dvdijcke/fdr/src/analysis/frontline_dwells_temporal_run.py", line 98, in <module>
    # test = model.run()
    # File "/home/dvdijcke/miniconda3/envs/fdd_new/lib/python3.9/site-packages/FDR/main.py", line 711, in run
    #     u_lower, u_upper, J_lower = self.conformalSplit()
    # File "/home/dvdijcke/miniconda3/envs/fdd_new/lib/python3.9/site-packages/FDR/main.py", line 530, in conformalSplit
    #     closest_grid_index = np.unravel_index(closest_index, (self.grid_x.shape[0], self.grid_x.shape[1]))
    # File "<__array_function__ internals>", line 180, in unravel_index
    # ValueError: index 24388 is out of bounds for array with size 24000


    test = model.run()
    file_name = 'frontline_dwells_temporal_run_9_51.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(test, file)
    