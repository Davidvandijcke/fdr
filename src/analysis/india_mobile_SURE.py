import pandas as pd 
import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import geopandas as gpd
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
from datetime import datetime


import ast
import matplotlib as mpl

if __name__ == '__main__':

    main_dir = "/home/dvdijcke"  #os.path.dirname(os.path.realpath(__file__))

    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    fn_merged = os.path.join(data_out, 'india', 'rajasatan_cheating_grid.geojson')
    gdf = gpd.read_file(fn_merged)
    

    # Load the shapefile data
    states_gdf = gpd.read_file(os.path.join(data_in, 'india', 'india_states_shapefile'))

    # Convert the CRS of the states GeoDataFrame to match the CRS of the Rajasthan GeoDataFrame
    states_gdf = states_gdf.to_crs(gdf.crs)

    # Extract the geometry for Rajasthan
    rajasthan_geometry = states_gdf[states_gdf['name_1'] == 'Rajasthan']['geometry'].values[0]

   # Reset the limits for the zoom (to show the entire Rajasthan)
    rajasthan_extent = states_gdf[states_gdf['name_1'] == 'Rajasthan'].total_bounds
    xlim = (rajasthan_extent[0], rajasthan_extent[2])
    ylim = (rajasthan_extent[1], rajasthan_extent[3])

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    # Plot the data
    vmax = 5
    p = gdf.plot(column="count_norm", cmap='coolwarm', vmax=vmax, alpha=0.7, ax=ax, legend=False)

    # Plot the Rajasthan border
    states_gdf.boundary.plot(color='k', linewidth=1, ax=ax, alpha=0.2)

    # Remove padding and axis
    ax.set_axis_off()

    # Set the plot limits to match the defined zoom area
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add a title to the plot

    # Create a custom colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=10)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
    fig.colorbar(cbar, ax=ax, orientation="vertical", pad=0)

    plt.show()
    