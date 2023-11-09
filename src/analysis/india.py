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
from shapely.geometry import mapping
from rasterio.mask import raster_geometry_mask, mask
import cv2
from FDD import FDD
from FDD.SURE import SURE
from pyproj import Transformer, CRS
from types import MethodType
import pickle
from datetime import datetime
from pyrosm import get_data
import pyrosm

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'


from pyspark import SparkContext
from pyspark.sql import SparkSession, SQLContext
from pyspark.sql import functions as F
from pyspark.sql.functions import pandas_udf, PandasUDFType, col, asc
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    ArrayType,
    FloatType,
    IntegerType,
    StringType,
)
from pyspark.sql import SQLContext, SparkSession
spark = SparkSession.builder.appName(f"spark").config("spark.driver.memory", "10g").config("spark.executor.memory", "10g").getOrCreate()
import ast
import matplotlib as mpl

def moveUp(fn, times = 1):
    for _ in range(times):
        fn = os.path.dirname(fn)
    return fn

def cast_to_grid(df, meters=5000):
    df = df.to_crs(epsg=3857)  # Project into Mercator (units in meters)

    # Get bounds
    minx, miny, maxx, maxy = df.geometry.total_bounds

    def create_grid(min_x, max_x, min_y, max_y, step):
        grid = []
        for x in np.arange(min_x, max_x, step):
            for y in np.arange(min_y, max_y, step):
                grid.append(Polygon([(x, y), (x+step, y), (x+step, y+step), (x, y+step)]))
        return grid

    # create the grid
    min_x, min_y, max_x, max_y = df.geometry.total_bounds
    grid = create_grid(min_x, max_x, min_y, max_y, meters) # 10km grid

    # create a GeoDataFrame from the grid
    grid_gdf = gpd.GeoDataFrame(geometry=grid)
    grid_gdf.crs = "EPSG:3857"

    # count the number of points in each grid cell
    count = gpd.sjoin(df, grid_gdf, how="inner", op="within").groupby("index_right").size()

    # join the counts back to the grid
    grid_gdf = grid_gdf.join(count.rename("count"), how="left")
    grid_gdf["count"] = grid_gdf["count"].fillna(0)
    
    return grid_gdf



if __name__ == '__main__':
    
    #-------
    # parameters
    six = False # after 6 or shutdown period?
    weekdy = "monday" # monday or sundays
    
    #-------
    if weekdy == "monday":
        cheatdate =  "2021-09-27"
        wkdy_suffix = "_monday"
    else:
        cheatdate =  "2021-09-26"
        wkdy_suffix = ""
        
        
    dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir)

    # get directory above
    main_dir = moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    

    data_out = os.path.join(main_dir, 'data', 'out')  
    ovrlf = "/Users/davidvandijcke/Dropbox (University of Michigan)/Apps/Overleaf/rdd/"
    tabs_dir = os.path.join(ovrlf, "tabs")
    figs_dir = os.path.join(ovrlf, "figs")
    
    if six:
        start = 18
        end = 24
        after6 = "_after6"
    else:
        start = 6
        end = 18
        after6 = ""
    
    redo_merge = False
    if redo_merge:
        #---
        # parameters
        fn_india = os.path.join(data_in, 'india', 'rajasatan_' + weekdy + '_sep_2021')
        #---
    
        df = spark.read.parquet(fn_india)
        
        
        ## day of cheating
        df = df.filter(F.to_date(df.date) == cheatdate) # day of cheating shutdown
        df = df.filter(F.hour(df.date).between(start,end)) # time of cheating shutdown
        
        fn = os.path.join(data_out, 'india', 'rajasatan_cheating' + after6 + wkdy_suffix)
        df.repartition(1).write.mode("overwrite").csv(fn, header=True, compression="gzip")
        
        # read csv file inside fn folder
        files_in_directory = os.listdir(fn)
        csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]

        cheat = pd.read_csv(*csv_files, compression="gzip")
        
        ## days before
        df = spark.read.parquet(fn_india)

        df = df.filter(F.to_date(df.date) != cheatdate) # day before cheating shutdown
        df = df.filter(F.hour(df.date).between(start,end)) # time of cheating shutdown
        
        fn = os.path.join(data_out, 'india', 'rajasatan_cheating_before' + after6 + wkdy_suffix)
        df.repartition(1).write.mode("overwrite").csv(fn, header=True, compression="gzip")
        
        # read csv file inside fn folder
        files_in_directory = os.listdir(fn)
        csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]
        before = pd.read_csv(*csv_files, compression="gzip")

        
        # convert to geopandas
        gdf = gpd.GeoDataFrame(cheat, geometry=gpd.points_from_xy(cheat.longitude, cheat.latitude))
        gdf = gdf.set_crs("epsg:4326")
        
        before = gpd.GeoDataFrame(before, geometry=gpd.points_from_xy(before.longitude, before.latitude))
        before = before.set_crs("epsg:4326")
        before = before.to_crs(epsg=3857)  # Project into Mercator (units in meters)


        
        #---- overlay grid onto data


        grid_gdf = cast_to_grid(gdf)
        
            
        grid_before = cast_to_grid(before) # need to divide by 3
        # aggregate
        
        
        # merge in before and divide    
        #temp = grid_gdf.sample(10000)
        merged = grid_gdf.sjoin(before, how="left", predicate="intersects").groupby("geometry").agg({"count": "mean", 'index_right' : 'count'}).reset_index()
        merged = merged.rename(columns={"index_right": "count_before"})

        
        merged["count_norm"] = merged["count"] / merged["count_before"]
        merged.loc[(merged['count'] == 0) | (merged['count_before'] == 0), 'count_norm'] = 0

        merged = gpd.GeoDataFrame(merged, geometry=merged.geometry)
        
        fn_merged = os.path.join(data_out, 'india', 'rajasatan_cheating_grid' + after6 + wkdy_suffix + '.geojson')
        merged.to_file(fn_merged, driver='GeoJSON')

    else:
        fn_merged = os.path.join(data_out, 'india', 'rajasatan_cheating_grid' + after6 + wkdy_suffix + '.geojson')
        gdf = gpd.read_file(fn_merged)
        gdf['count_norm'] = gdf['count_norm'] * 100

    


    # Load the shapefile data
    states_gdf = gpd.read_file(os.path.join(data_in, 'india', 'india_states_shapefile'))

    # Convert the CRS of the states GeoDataFrame to match the CRS of the Rajasthan GeoDataFrame
    states_gdf = states_gdf.to_crs(gdf.crs)
    

    # Extract the geometry for Rajasthan
    rajasthan_geometry = states_gdf[states_gdf['name_1'] == 'Rajasthan']['geometry']

   # Reset the limits for the zoom (to show the entire Rajasthan)
    rajasthan_extent = states_gdf[states_gdf['name_1'] == 'Rajasthan'].total_bounds
    xlim = (rajasthan_extent[0], rajasthan_extent[2])
    ylim = (rajasthan_extent[1], rajasthan_extent[3])


    #---------
    # Figure: raw data 
    #--------
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10,8))

    # Plot the data
    vmax = 200
    im = gdf.plot(column="count_norm", cmap='coolwarm', vmin=0, vmax=vmax, ax=ax, alpha=0.9)

    # Plot the Rajasthan border
    rajasthan_geometry.boundary.plot(color='k', linewidth=2, ax=ax, alpha=0.5)



    # Create a mappable for colorbar using ScalarMappable
    norm = plt.Normalize(vmin=0, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap='coolwarm')
    sm.set_array([]) # Need to set_data with an empty array


    # Position and add the colorbar
    cbar_ax = fig.add_axes([0.1, -0.1, 0.8, 0.03])  # [left, bottom, width, height]
    cbar_ax.set_title(r"Index (Average on Previous 3 Sundays=100)", fontsize=22, loc = "center")
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.axvline(100, color='k')  # Add vertical line at 100
    cbar.ax.tick_params(labelsize=18)
    cbar.outline.set_visible(False)


    # Remove padding and axis
    ax.set_axis_off()
    ax.set_rasterized(True)
    ax.margins(0)
    ax.axis("equal")


    # Set the plot limits to match the defined zoom area
    plt.tight_layout()

    plt.savefig(os.path.join(figs_dir, 'india_raw' + after6 + wkdy_suffix + '.pdf'), dpi=300, bbox_inches='tight')
    
    
    #---------
    # Figure: satellite view
    #--------
    
    def plotSatellite(provider = ctx.providers.Esri.WorldImagery):
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(10,8))

        # Plot the Rajasthan border
        rajasthan_geometry.boundary.plot(color='k', linewidth=1, ax=ax, alpha=0.5, rasterized=True)
        
        ctx.add_basemap(ax, crs=gdf.crs.to_string(), source=provider,  attribution_size=5)
        
        ax.set_axis_off()
        ax.margins(0)
        #ax.axis("equal")
        
    plotSatellite()
    plt.savefig(os.path.join(figs_dir, 'india_satellite_imagery.pdf'), bbox_inches='tight')
    
    plotSatellite(provider=ctx.providers.OpenStreetMap.Mapnik)
    plt.savefig(os.path.join(figs_dir, 'india_satellite_osm.pdf'), bbox_inches='tight')
     
    for provider in ctx.providers:
        print(provider)
        

