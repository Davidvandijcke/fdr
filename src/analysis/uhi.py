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
from shapely.geometry import mapping
from rasterio.mask import raster_geometry_mask, mask
import cv2
from FDD import FDD
from pyproj import Transformer, CRS

from types import MethodType

def readSatellite(cname):
        #------- read satellite data
    # https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-1619_Landsat8-9-Collection2-Level2-Science-Product-Guide-v5.pdf

    # Open the ST_B10 and QA bands
    # folder = os.path.join(data_in, 'satellite', cname)
    fn = os.path.join(cname + "_ST_B10.TIF")
    with rasterio.open(fn) as src:
        st_band = src.read(1)  # Note: band indexing in rasterio is 1-based
        st_transform = src.transform
        st_crs = src.crs

    # set fill value -9999 to nan
    st_band = np.where(st_band == 0, np.nan, st_band)

    # filter out values where st_band not between 293 and 61440
    st_band = np.where(st_band < 293, np.nan, st_band)
    st_band = np.where(st_band >= 61440, np.nan, st_band)

    # ST_QA band contains the uncertainty of the ST band, in Kelvin.
    fn = os.path.join(cname + "_ST_QA.TIF")
    with rasterio.open(fn) as src:
        qa_band = src.read(1)

    # Apply scale factor to convert digital numbers to temperature in Kelvin
    st_k = st_band * 0.00341802 + 149

    qa_band = np.where(qa_band == 0, np.nan, st_band)

    qa_band = qa_band * 0.01 # https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1328_Landsat8-9-OLI-TIRS-C2-L2-DFCB-v6.pdf p.5

    # Convert Kelvin to Celsius
    st_c = st_k - 273.15

    # Define a threshold for QA values
    qa_threshold = 2  # max 2 degrees Kelvin uncertainty

    # Apply QA mask to ST band
    # leaving this for now, it's something annoying with bit-packed numbers
    #st_c_masked = np.where(qa_band < qa_threshold, st_c, np.nan)
    return st_c, qa_band, st_transform, st_crs

def plotHeatMap(minx, maxx, miny, maxy):

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
    gdf.plot(ax=ax, facecolor="none", edgecolor='black', alpha=0.5)

    # add basemap   
    ctx.add_basemap(ax, crs=st_crs, alpha=0.5, source=ctx.providers.Stamen.TonerLite)
    
    return fig, ax

def subsetRaster(cname, minx, maxx, miny, maxy, st_crs):
    
    folder = os.path.join(cname)
    fn = os.path.join(folder, cname + "_ST_B10.TIF")
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
        
        
    # get relative dir
    #dir = os.path.dirname(__file__)

    # get directory above
    data_in = "/home/dvdijcke/data/fdd/"

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
    
    # bounding box around sacramento
    minx, maxx = -121.787773, -121.037523 # Longitude
    miny, maxy = 38.246354, 38.922722 # Latitude

    fig, ax = plotHeatMap(minx, maxx, miny, maxy)

    # subset the raster on sacramento area and also on heat island area
    # bounding box around sacramento
    minx, maxx = -121.787773, -121.237523 # Longitude
    miny, maxy = 38.246354, 38.922722 # Latitude
    subset = subsetRaster(cname, minx, maxx, miny, maxy, st_crs)




    fig, ax = plt.subplots(figsize=(8, 8))
    ret = rasterio.plot.show(subset, cmap='RdYlBu_r', ax=ax)
    
    imgseg = cv2.resize(subset.squeeze(), dsize = (0,0), fx=0.05, fy=0.05)
    plt.imshow(imgseg)



    # lower the resolution of the subset image
    # https://rasterio.readthedocs.io/en/latest/topics/resampling.html
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling
    # https://rasterio.readthedocs.io/en/latest/topics/resampling.html#resampling
    # https://rasterio.readthedocs.io/en/latest/topics/resampling.html#resampling
    # https://rasterio.readthedocs.io/en/latest/topics/resampling.html#resampling
    
    
    # segment the image

    # def castDataToGridSmooth(self):
        
    #     n = self.Y.shape[0]
        
    #     if self.resolution is None:
    #         self.resolution = 1/int(np.sqrt(n)) # int so we get a natural number of grid cells
        
    #     xmax = np.max(self.X, axis = 0)
        
    #     # set up grid
    #     grid_x = np.meshgrid(*[np.arange(0, xmax[i], self.resolution) for i in range(self.X.shape[1])])
    #     grid_x = np.stack(grid_x, axis = -1)
    #     if self.Y.ndim > 1: # account for vector-valued outcomes
    #         grid_y = np.zeros(list(grid_x.shape[:-1]) + [self.Y.shape[1]])
    #     else:
    #         grid_y = np.zeros(list(grid_x.shape[:-1]))
    #     grid_x_og = np.empty(list(grid_x.shape[:-1]), dtype = object) # assign original x values as well for later
        
    #     # Get the indices of the grid cells for each data point
    #     indices = [(np.clip((self.X[:, i]) // self.resolution, 0, grid_y.shape[i] - 1)).astype(int) for i in reversed(range(self.X.shape[1]))]
    #     indices = np.array(indices).T

    #     # Create a count array to store the number of data points in each cell
    #     counts = np.zeros_like(grid_y)

    #     # Initialize grid_x_og with empty lists
    #     for index in np.ndindex(grid_x_og.shape):
    #         grid_x_og[index] = []
        

    #     # Iterate through the data points and accumulate their values in grid_y and grid_x_og
    #     for i, index_tuple in enumerate(indices):
    #         index = tuple(index_tuple)
    #         print(index)
    #         if np.all(index < grid_y.shape):
    #             # add  Y value to grid cell
    #             # print(index)
    #             # print(i)
    #             grid_y[index] += self.Y[i]
    #             counts[index] += 1
    #             grid_x_og[index].append(self.X[i])
        
        

    #     # Divide the grid_y by the counts to get the average values
    #     grid_y = np.divide(grid_y, counts, where=counts != 0, out=grid_y)

    #     # Find the closest data point for empty grid cells
    #     empty_cells = np.where(counts == 0)
    #     empty_cell_coordinates = np.vstack([empty_cells[i] for i in range(self.X.shape[1])]).T * self.resolution
    #     if empty_cell_coordinates.size > 0:
    #         tree = cKDTree(self.X + self.resolution / 2) # get centerpoints of hypervoxels
    #         _, closest_indices = tree.query(empty_cell_coordinates, k=1)
    #         closest_Y_values = self.Y[closest_indices]

    #         # Assign the closest data point values to the empty grid cells
    #         grid_y[empty_cells] = closest_Y_values
        
    #     # add an extra "channel" dimension if we have a scalar outcome
    #     if self.Y.ndim == 1:
    #         grid_y = grid_y.reshape(grid_y.shape + (1,))

    #     self.grid_x_og = grid_x_og
    #     self.grid_x = grid_x
    #     self.grid_y = grid_y
        
        
    X = np.stack([np.tile(np.arange(0, imgseg.shape[0], 1), imgseg.shape[1]), 
              np.repeat(np.arange(0, imgseg.shape[0], 1), imgseg.shape[1])], axis = 1)
    Y = imgseg.flatten()
    
    
    model = FDD(Y=Y, X = X, level = 16, lmbda = 100, nu = 0.01, iter = 10000, tol = 5e-5, 
        pick_nu = "MS", scaled = True, scripted = False, image=False, rectangle=True)
    
    
    u, jumps, J_grid, nrj, eps, it = model.run()
