import pandas as pd 
import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import box, Polygon, Point
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
import contextily as ctx
from matplotlib import pyplot as plt
from shapely.geometry import mapping
from rasterio.mask import raster_geometry_mask, mask
import cv2
from FDD import FDD
from FDD.SURE import SURE
from pyproj import Transformer, CRS
from types import MethodType
import pickle
import boto3
import ray

from types import MethodType


#----- parameters
downsize = 0.25
S=32
# # bounding box around sacramento
# minx, maxx = -121.787773, -121.037523 # Longitude
# miny, maxy = 38.246354, 38.922722 # Latitude

# tighter bounding box around Chicago's urban core
minx, maxx = -87.884935, -87.579413 # Longitude
miny, maxy = 41.687979, 41.979697 # Latitude

minx, maxx = -87.884935, -87.500000 # Longitude
miny, maxy = 41.687979, 41.979697 # Latitude

# bounding box around specified area in Houston
# slightly zoomed in bounding box around specified area in Houston
minx, maxx = -95.512047, -95.257086 # Longitude
miny, maxy = 29.666534, 29.867654 # Latitude






#------
def readSatellite(cname):
    #------- read satellite data
    # https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v5.pdf

    # Open the ST_B10 and QA bands
    # folder = os.path.join(data_in, 'satellite', cname)
    fn = os.path.join(cname)
    with rasterio.open(fn) as src:
        st_band = src.read(1)  # Note: band indexing in rasterio is 1-based
        st_transform = src.transform
        st_crs = src.crs
        st_meta = src.meta

    # set fill value -9999 to nan
    st_band = np.where(st_band == 0, np.nan, st_band)

    # filter out values where st_band not between 293 and 61440
    st_band = np.where(st_band < 293, np.nan, st_band)
    st_band = np.where(st_band >= 61440, np.nan, st_band)

    # # ST_QA band contains the uncertainty of the ST band, in Kelvin.
    # fn = os.path.join(cname + "_QA.TIF")
    # with rasterio.open(fn) as src:
    #     qa_band = src.read(1)

    # Apply scale factor to convert digital numbers to temperature in Kelvin
    st_k = st_band * 0.00341802 + 149

    # qa_band = np.where(qa_band == 0, np.nan, st_band)

    # qa_band = qa_band * 0.01 # https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1328_Landsat8-9-OLI-TIRS-C2-L2-DFCB-v6.pdf p.5

    # Convert Kelvin to Celsius
    st_c = st_k - 273.15

    # # Define a threshold for QA values
    # qa_threshold = 2  # max 2 degrees Kelvin uncertainty

    # Apply QA mask to ST band
    # leaving this for now, it's something annoying with bit-packed numbers
    #st_c_masked = np.where(qa_band < qa_threshold, st_c, np.nan)
    return st_c, st_transform, st_crs

def merge_rasters(raster1_path, raster2_path, output_path):
    # Open the rasters
    with rasterio.open(raster1_path) as src1, rasterio.open(raster2_path) as src2:
        # Merge the rasters
        out_image, out_transform = merge([src1, src2])
        
        # Get the metadata from the first file
        meta = src1.meta.copy()
        
        # Update the metadata with new dimensions, transform (affine)
        meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform,
        })

        # Write the merged raster to a new file
        with rasterio.open(output_path, "w", **meta) as dest:
            dest.write(out_image)

def plotHeatMap(minx, maxx, miny, maxy, st_c_masked, st_transform, st_crs, alpha = 0.5):
    # use alpha to control whether we see the surface temperature or not

    # create a polygon from the bounding box    
    bbox = box(minx, miny, maxx, maxy)

    # create a geodataframe with the polygon
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="epsg:4326")
    gdf = gdf.to_crs(st_crs)

    extent = [gdf.bounds.minx[0], gdf.bounds.maxx[0], gdf.bounds.miny[0], gdf.bounds.maxy[0]]

    fig, ax = plt.subplots(figsize=(12, 12))
    ret = rasterio.plot.show(st_c_masked, cmap='RdYlBu_r', ax=ax, transform=st_transform)
    img = ret.get_images()[0]
    fig.colorbar(img, ax=ax, fraction=.05)
    
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # plot the geodataframe on top of the raster
    gdf.plot(ax=ax, facecolor="none", edgecolor='black', alpha=0.2)

    # add basemap   
    ctx.add_basemap(ax, crs=st_crs, alpha=alpha, source=ctx.providers.OpenStreetMap.Mapnik)
    
    return fig, ax

def subsetRaster(cname, minx, maxx, miny, maxy, st_crs):
    
    folder = os.path.join(cname)
    fn = os.path.join(folder, cname)
    # Define your bounding box and create a polygon
    # Define points in the old CRS
    lower_left = Point(minx, miny)
    upper_right = Point(maxx, maxy)

    # Define a transformer from the old CRS to the new CRS
    transformer = Transformer.from_crs(CRS.from_epsg(4326), st_crs, always_xy=True)

    # Transform the points to the new CRS
    lower_left_transformed = transformer.transform(*lower_left.coords[0])
    upper_right_transformed = transformer.transform(*upper_right.coords[0])

    # create the bounding box in the new CRS
    bbox = box(*lower_left_transformed, *upper_right_transformed)

    # create a GeoDataFrame with the polygon
    gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=st_crs)
    
    # Transform the GeoDataFrame to a format that can be used by rasterio
    geoms = gdf.geometry.values.tolist()
    geometry = [mapping(geoms[0])]
    
    # Use the mask function to clip the raster array
    with rasterio.open(fn) as src:
        st_c_masked, out_transform = mask(src, geometry, crop=True, filled=True, all_touched=True)
    
    st_c_masked = np.where(st_c_masked == 0, np.nan, st_c_masked)


    return st_c_masked




if __name__ == '__main__':
    
    
    # get directory above
    data_in = "/home/dvdijcke/data/in/"
    
    # combine all the tifs into one
    sn = "LC09_CU_021007_20220617_20230414_02_ST" # landsat ARD surface temperature
    raster1_path = os.path.join(data_in, "satellite", sn, sn + "_B10.TIF") 
    
    sn = "LC09_CU_021008_20220617_20230414_02_ST"
    raster2_path = os.path.join(data_in, "satellite", sn, sn + "_B10.TIF")
    
    output_path = os.path.join(data_in, "satellite", "merged.TIF")

    merge_rasters(raster1_path, raster2_path, output_path)

    
    
    st_c_masked, st_transform, st_crs = readSatellite(output_path)
    

    
    ray.init()
        
    # get relative dir
    #dir = os.path.dirname(__file__)


    # cname = "LC08_L2SP_044033_20220628_20220706_02_T1/"
    
    # # pull tif file from s3
    # s3 = boto3.resource('s3')
    # bucket = s3.Bucket('ipsos-dvd')
    # for obj in bucket.objects.filter(Prefix='fdd/data/in/satellite/LC08_L2SP_044033_20220628_20220706_02_T1/'):
    #     print(obj.key)
    #     bucket.download_file(obj.key, os.path.join(data_in, 'satellite', obj.key.split('/')[-1]))
    
    sn = "LC08_L2SP_044033_20220628_20220706_02_T1"
    cname = os.path.join(data_in, "satellite", sn, sn)
    st_c_masked, qa_band, st_transform, st_crs = readSatellite(cname)

    # # Bounding box around Austin, Texas, metro area
    # # Bounding box coordinates for Austin, Texas metropolitan area
    # minx = -98.345211
    # miny = 29.705602
    # maxx = -96.848323
    # maxy = 30.740314

    # # Bounding box around Houston, Texas, metro area
    # # # Bounding box coordinates for Austin, Texas metropolitan area
    # minx, maxx = -95.823268, -95.069705 # Longitude
    # miny, maxy = 29.523624, 30.110731 # Latitude
    
    # # bounding box around Chicago, Illinois, metro area
    # minx, maxx = -88.140101, -87.524137 # Longitude
    # miny, maxy = 41.444543, 42.348038 # Latitude
    


    fig, ax = plotHeatMap(minx, maxx, miny, maxy, st_c_masked, st_transform, st_crs)

    # subset the raster on sacramento area and also on heat island area
    subset = subsetRaster(cname, minx, maxx, miny, maxy, st_crs)


    fig, ax = plt.subplots(figsize=(8, 8))
    ret = rasterio.plot.show(subset, cmap='RdYlBu_r', ax=ax)
    
    imgseg = cv2.resize(subset.squeeze(), dsize = (0,0), fx=downsize, fy=downsize)
    plt.imshow(imgseg)

    X = np.stack([np.tile(np.arange(0, imgseg.shape[0], 1), imgseg.shape[1]), 
              np.repeat(np.arange(0, imgseg.shape[0], 1), imgseg.shape[1])], axis = 1)
    Y = imgseg.flatten()
    
    resolution = 1/int(np.sqrt(2/3*Y.size))

    model = FDD(Y=Y, X = X, level = S, lmbda = 5, nu = 0.001, iter = 10000, tol = 5e-5, 
        pick_nu = "MS", scaled = True, scripted = False, image=False, rectangle=True, resolution=resolution)
    num_samples =  400 #  400 # 400 # 400 # 200
    R =  3 # 3 # 3 # 5
    num_gpus = 1
    num_cpus = 4
    res = SURE(tuner=True, num_samples=num_samples, model=model, R=R, 
        num_gpus=num_gpus, num_cpus=num_cpus)

    file_name = 'uhi_SURE.pkl'
    with open(file_name, 'wb') as file:
        pickle.dump(res, file)
    
    # s3 = boto3.client('s3')
    # with open(file_name, "rb") as f:
    #     s3.upload_fileobj(f, "ipsos-dvd", "fdd/data/uhi_SURE.pkl")

    
    # u, jumps, J_grid, nrj, eps, it = model.run()
    
    # u_scaled = u / np.max(model.Y_raw, axis = -1)
    

    # plt.imsave("test.png", u)

# file_name = 'uhi_SURE.pkl'
# with open(file_name, 'rb') as file:
#     res = pickle.load(file)
# print(res)