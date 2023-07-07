import pandas as pd 
import os

os.environ['USE_PYGEOS'] = '0'
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import box, Polygon, Point
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import contextily as ctx
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import mapping
from rasterio.mask import raster_geometry_mask, mask
import cv2
from FDD import FDD
from FDD.SURE import SURE
from pyproj import Transformer, CRS
from types import MethodType
import pickle
import boto3

from uhi import *


def plotMap(minx, maxx, miny, maxy, st_crs, source = ctx.providers.Esri.WorldImagery):
    # use alpha to control whether we see the surface temperature or not

    # create a polygon from the bounding box    
    bbox = box(minx, miny, maxx, maxy)


    # create a geodataframe with the polygon
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="epsg:4326")
    gdf = gdf.to_crs(st_crs)

    fig, ax = plt.subplots(figsize=(8, 8))

    # plot the geodataframe on top of the raster
    gdf.plot(ax=ax, facecolor="none", edgecolor='black', alpha=0)

    # add basemap   
    ctx.add_basemap(ax, crs=st_crs, source=source, attribution_size=2)
    
    # Creates a "blank" colorbar placeholder to line up with the LST map
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(ax.collections[0], cax=cax)
    cbar.set_label('°C', rotation=0, labelpad=10, alpha=0)
    cbar.set_ticks([])
    cbar.outline.set_visible(False)
    
    # make plot prettier
    ax.set_axis_off()
    
    return fig, ax

def plotLST(imgseg):
    # apply scaling factor and convert to celsius
        
    # Assuming imgseg is an array-like image data and figs_dir is a directory path
    fig, ax = plt.subplots(figsize=(8, 8)) # you can adjust the size of the figure to your liking
    im = ax.imshow(imgseg, cmap='RdYlBu_r') # image display

    # Create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    plt.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label('°C', rotation=0, labelpad=10) # This adds '°C' label to colorbar
    

    # Improving layout and saving
    plt.tight_layout()
    
    return fig, ax

if __name__ == '__main__':
              
    #---------------------------
    # get data and plot maps
    #---------------------------
    
    # paths
    dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir)
    from uhi import *
    from simulations_2d_postprocessing import *

    # get directory above
    main_dir = moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    # overleaf synced dropbox folder (change to your own overleaf path)
    figs_dir = "/Users/davidvandijcke/Dropbox (University of Michigan)/Apps/Overleaf/rdd/figs/"
    
    # get satellite data
    
    #cname = "LC09_CU_024007_20220621_20230414_02_ST" # detroit
    #cname=  "LC08_CU_017017_20220623_20220709_02_ST" # houston "LC08_L2SP_044033_20220628_20220706_02_T1"
    cname = "LC09_CU_024013_20220621_20230414_02_ST" #atlanta
    fn = os.path.join(data_in, 'satellite', "atlanta", cname, cname + "_B10.TIF")
    #fn = os.path.join(data_in, "satellite", "chicago_june17_2022_merged.TIF")

    st_c_masked, st_transform, st_crs = readSatellite(fn)

    # plot streetmap
    fig, ax = plotMap(minx, maxx, miny, maxy, st_crs, source = ctx.providers.Esri.WorldStreetMap)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "uhi_streetmap.pdf"), dpi = 300, bbox_inches='tight')

    # plot satellite map
    fig, ax = plotMap(minx, maxx, miny, maxy, st_crs, source = ctx.providers.Esri.WorldImagery)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "uhi_satellite.pdf"), dpi = 300, bbox_inches='tight')
    

    # subset the raster on sacramento area and also on heat island area
    subset = subsetRaster(fn, minx, maxx, miny, maxy, st_crs)

    # downsize image for faster processing
    fig, ax = plt.subplots(figsize=(8, 8))
    ret = rasterio.plot.show(subset, cmap='RdYlBu_r', ax=ax)
    imgseg = cv2.resize(subset.squeeze(), dsize = (0,0), fx=downsize, fy=downsize)
    
    imgseg = (imgseg * 0.00341802 + 149) - 273.15
    
    #plt.hist(imgseg)
    
    bottom = np.quantile(imgseg, 0.01) # bottom quantile without Lake Michigan
    top = np.quantile(imgseg, 0.99)
    
    imgseg[imgseg < bottom] = bottom
    imgseg[imgseg > top] = top
    
    # plot land surface temperature map
    plotLST(imgseg)
    plt.savefig(os.path.join(figs_dir, "uhi_to_segment.pdf"), dpi = 300, bbox_inches='tight')




    #---------------------------
    # segment image
    #---------------------------

    # read the pkl file
    file_name = 'uhi_SURE.pkl'
    
    with open(file_name, 'rb') as file:
        res = pickle.load(file)
        
    # get lmbda, nu from res
    best = res.get_best_result(metric = "score", mode = "min")

    config = best.metrics['config']
    lmbda, nu = config['lmbda'], config['nu']
    
    lmbda = 100
    nu = 0.01
    
    # segment the image

    X = np.indices(imgseg.shape)
    # flatten last two dimensions of X
    X = X.reshape((X.shape[0], -1)).T
    Y = imgseg.flatten()
    
    resolution = 1/int(np.sqrt(1/2*Y.size))
    model = FDD(Y=Y, X = X, level = S, lmbda = lmbda, nu = nu, iter = 10000, tol = 5e-5, 
        pick_nu = "MS", scaled = True, scripted = False, image=False, rectangle=True, resolution=resolution)
    num_samples = 225 #  400 # 400 # 400 # 200
    R =  3 # 3 # 3 # 5
    num_gpus = 0.5
    num_cpus = 4
    res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
        num_gpus=num_gpus, num_cpus=num_cpus)

    file_name = 'uhi_SURE.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(res, file)
    
    s3 = boto3.client('s3')
    with open(file_name, "rb") as f:
        s3.upload_fileobj(f, "ipsos-dvd", "fdd/data/uhi_SURE.pkl")

    
    u, jumps, J_grid, nrj, eps, it = model.run()
    
    # u_scaled = u / np.max(model.Y_raw, axis = -1)
    

    # plt.imsave("test.png", u)