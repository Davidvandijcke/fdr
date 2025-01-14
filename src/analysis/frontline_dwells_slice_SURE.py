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

def getSpaceDwells(week, grid):
    """Get the spatio-temporal data for the frontline dwells
    Args:
        start (int): first week of 2022 to consider
        end (int): last week of 2022 to consider
    """
    
    # assign period matrices
    agg_gdf = grid[grid.week==week]
    X1 = np.array(agg_gdf.geometry.centroid.x)
    X2 = np.array(agg_gdf.geometry.centroid.y)
    X = np.stack([X2, X1]).T 
    Y_t = np.array(agg_gdf['count_ratio'])

    # winsorize
    qtile = np.quantile(Y_t, 0.95)
    Y_t[Y_t > qtile] = qtile # 0.5
    
    return(X, Y_t)


if __name__ == "__main__":

    #----------------
    # parameters
    #----------------
    S = 16
    N = 1000
    lmbda = 50
    nu = 0.05
    num_samples = 144 # 225 #  400 # 400 # 400 # 200
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
    
    grid = gpd.read_file(data_out + "grid_dwells_weekly.geojson")
    grid = grid.set_crs("EPSG:6381", allow_override=True)
    
    for week in range(9, 52):
        X_t, Y_t = getSpaceDwells(week, grid)

        # segment
        #--------

        lmbda, nu = 50, 0.05 # just to init model

        grid_n = np.array([len(np.unique(X_t[:,0])), len(np.unique(X_t[:,1]))])
        model = FDR(Y_t, X_t, level = S, lmbda = lmbda, nu = nu, iter = iter, tol = 5e-5, pick_nu = "MS", 
                    CI=False, rectangle=False, grid_n=grid_n, scripted=False) 

        # test = model.run()
        # file_name = 'frontline_dwells_temporal_SURE.pkl'
        # with open(file_name, 'wb') as file:
        #     pickle.dump(test, file)
        
        res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
            num_gpus=num_gpus, num_cpus=num_cpus, lmbda_max=lmbda_max,
            nu_min=nu_min, nu_max=nu_max)

        file_name = '/home/dvdijcke/data/out/dwells_slice/frontline_dwells_slice_SURE_' + str(week) + '_.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(res, file)