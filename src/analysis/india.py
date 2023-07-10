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
from pyrosm import get_data
import pyrosm


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

def moveUp(fn, times = 1):
    for _ in range(times):
        fn = os.path.dirname(fn)
    return fn

def cast_to_grid(df):
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
    grid = create_grid(min_x, max_x, min_y, max_y, 5000) # 1km grid

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
    
    #spend_list = [spend_dir + "m=" + str(x) + "/" for x in [9,10]]
    
    dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir)

    # get directory above
    main_dir = moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    # df = spark.read.parquet(os.path.join(data_in, 'india', 'rajasatan_sep_25_30_2021'))
    
    # ## day of cheating
    # df = df.filter(F.to_date(df.date) == "2021-09-26") # day of cheating shutdown
    # df = df.filter(F.hour(df.date).between(6,18)) # time of cheating shutdown
    
    # fn = os.path.join(data_out, 'india', 'rajasatan_cheating')
    # df.repartition(1).write.mode("overwrite").csv(fn, header=True, compression="gzip")
    
    # # read csv file inside fn folder
    # files_in_directory = os.listdir(fn)
    # csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]

    # cheat = pd.read_csv(*csv_files, compression="gzip")
    
    # ## day before
    # df = spark.read.parquet(os.path.join(data_in, 'india', 'rajasatan_sep_25_30_2021'))

    # df = df.filter(F.to_date(df.date) == "2021-09-25") # day before cheating shutdown
    # df = df.filter(F.hour(df.date).between(6,18)) # time of cheating shutdown
    
    # fn = os.path.join(data_out, 'india', 'rajasatan_cheating_before')
    # df.repartition(1).write.mode("overwrite").csv(fn, header=True, compression="gzip")
    
    # # read csv file inside fn folder
    # files_in_directory = os.listdir(fn)
    # csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]
    # before = pd.read_csv(*csv_files, compression="gzip")

    
    
    # # convert to geopandas
    # gdf = gpd.GeoDataFrame(cheat, geometry=gpd.points_from_xy(cheat.longitude, cheat.latitude))
    # gdf = gdf.set_crs("epsg:4326")
    
    # before = gpd.GeoDataFrame(before, geometry=gpd.points_from_xy(before.longitude, before.latitude))
    # before = before.set_crs("epsg:4326")
    # before = before.to_crs(epsg=3857)  # Project into Mercator (units in meters)


    
    # #---- overlay grid onto data


    # grid_gdf = cast_to_grid(gdf)
    
        
    # grid_before = cast_to_grid(before)
    
    # # merge in before and divide    
    # #temp = grid_gdf.sample(10000)
    # merged = grid_gdf.sjoin(before, how="left", predicate="intersects").groupby("geometry").agg({"count": "mean", 'index_right' : 'count'}).reset_index()
    # merged = merged.rename(columns={"index_right": "count_before"})

    
    # merged["count_norm"] = merged["count"] / merged["count_before"]
    # merged.loc[(merged['count'] == 0) | (merged['count_before'] == 0), 'count_norm'] = 0

    # merged = gpd.GeoDataFrame(merged, geometry=merged.geometry)
    
    # merged.to_file(os.path.join(data_out, 'india', 'rajasatan_cheating_grid.geojson'), driver='GeoJSON')

    # fig, ax = plt.subplots(figsize=(10, 10))
    
    # p = merged.plot(column="count_norm", legend=True, vmax = 10, alpha=0.7, ax=ax)
    # # remove padding and axis
    # ax.set_axis_off()
    
    # # Get the extent of the polygon
    # polygon_extent = merged.total_bounds

    # # Set the plot limits to match the extent of the polygon
    # ax.set_xlim(polygon_extent[0], polygon_extent[2])
    # ax.set_ylim(polygon_extent[1], polygon_extent[3])
    
    # # make colorbar smaller
    
    
    
    # # add basemap
    # ctx.add_basemap(p, source=ctx.providers.Stamen.TonerLite)
    # #plt.axis("off")
    # #plt.margin(0)
    
    
    
    #### calculate dwells in shops
    fn = os.path.join(data_out, 'india', 'rajasatan_cheating')

    # read csv file inside fn folder
    files_in_directory = os.listdir(fn)
    csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]
    before = pd.read_csv(*csv_files, compression="gzip")
    
    before = gpd.GeoDataFrame(before, geometry=gpd.points_from_xy(before.longitude, before.latitude))
    
    # get before bounding box
    bbox = before.total_bounds
    

    reload_osm = False
    
    fn_json = os.path.join(data_out, 'india', 'shops_india.geojson')


    if reload_osm:
        # get India data
        # fp = get_data("India")
        
        fn = os.path.join(data_in, 'india', 'india-220101.osm.pbf')
        fn_osm = os.path.join(data_in, 'india', 'india-220101.osm')
        fn_shops = os.path.join(data_out, 'india', 'shops_india.osm')

        
        # Separate out the bounding box values
        left = bbox[0]
        bottom = bbox[1]
        right = bbox[2]
        top = bbox[3]

        # Construct the osmosis command
        osmosis_command = f"osmosis --read-pbf '{fn}' --bounding-box top={top} left={left} bottom={bottom} right={right} --write-xml '{fn_osm}'"
        
        os.system(osmosis_command)

        osmfilter_command = f"osmfilter '{fn_osm}' --keep=\"shop=*\" -o='{fn_shops}'"

        # Execute the command
        os.system(osmfilter_command)
        
        osmconvert_command = f"osmconvert '{fn_shops}' -o='{fn_shops+'.pbf'}'"
        os.system(osmconvert_command)


        osmium_command = f"osmium export '{fn_shops}' -o '{fn_json}'"
        os.system(osmium_command)

    else:
        shops = gpd.read_file(fn_json)
        shops = shops.to_crs("epsg:3587")
        p = shops.plot()
        ctx.add_basemap(p, source=ctx.providers.Stamen.TonerLite)
        
    # plot bbox
    