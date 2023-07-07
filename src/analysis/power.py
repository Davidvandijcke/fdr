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
from datetime import datetime

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


from types import MethodType

def moveUp(fn, times = 1):
    for _ in range(times):
        fn = os.path.dirname(fn)
    return fn


def filterPlaces(places_dir, sqlContext, minx, maxx, miny, maxy):
    ## load places
    
    places =  (sqlContext.read.format('com.databricks.spark.csv')
         .options(header='true', inferschema='true')
         .option('escape','"') # this is necessary because the fields contain "," as well but dont have quotes around them
         .load(places_dir))

    # filter on bounding box
    places = places.filter((F.col("longitude") > minx) & (F.col("longitude") < maxx) & (F.col("latitude") > miny) & (F.col("latitude") < maxy))
    
    places = places.cache()
    places.count()
    
    places = places.toPandas()
    
    return places

if __name__ == '__main__':
    
    #spend_list = [spend_dir + "m=" + str(x) + "/" for x in [9,10]]
    
    dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir)

    # get directory above
    main_dir = moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    sqlContext = SQLContext(spark)

    ### load spend
    # spend =  (sqlContext.read.format('com.databricks.spark.csv')
    #      .options(header='true', inferschema='true')
    #      .option('escape','"') # this is necessary because the fields contain "," as well but dont have quotes around them
    #      .load(spend_dir))
    
    # str_to_list = F.udf(lambda row: ast.literal_eval(row), ArrayType(FloatType()))
    # spend = spend.withColumn("list", str_to_list(spend["spend_by_day"]))
    
    # bounding box around Houston, Texas
    minx, maxx = -95.823268, -95.069705 # Longitude
    miny, maxy = 29.523105, 30.110731 # Latitude
    
    # Detroit
    minx, maxx = min(-83.082778, -83.110556, -83.031389, -83.003889, -83.082778), max(-83.082778, -83.110556, -83.031389, -83.003889, -83.082778) # Longitude
    miny, maxy = min(42.315833, 42.372222, 42.393333, 42.337222, 42.315833), max(42.315833, 42.372222, 42.393333, 42.337222, 42.315833) # Latitude



    
    ### load places data for city
    redo_places = False
    if redo_places:
        places_dir = "/Users/davidvandijcke/Dropbox (University of Michigan)/dynamic/repo/data/places/2022/"
        

        places = filterPlaces(places_dir, sqlContext, minx, maxx, miny, maxy)
        
        places.to_csv(os.path.join(data_out, "places_detroit.csv.gz"))
    else:
        places = pd.read_csv(os.path.join(data_out, "places_detroit.csv.gz"))
        
        
    ### load patterns data for city
    pat_dir = ["/Users/davidvandijcke/Dropbox (University of Michigan)/dynamic/repo/data/patterns/2022/6/", "/Users/davidvandijcke/Dropbox (University of Michigan)/dynamic/repo/data/patterns/2022/5/",
               "/Users/davidvandijcke/Dropbox (University of Michigan)/dynamic/repo/data/patterns/2022/4/",]
    pat = filterPlaces(pat_dir, sqlContext, minx, maxx, miny, maxy)
    pat.to_csv(os.path.join(data_out, "patterns_atlanta_4_6_2022.csv.gz"))
    
    pat = pd.read_csv(os.path.join(data_out, "patterns_atlanta_4_6_2022.csv.gz"))
    
    # first two digits of naics code
    
    # houston hunter creek village
    # minx, maxx = min(-95.578611, -95.581389, -95.446111, -95.442222, -95.443611, -95.578611), max(-95.578611, -95.581389, -95.446111, -95.442222, -95.443611, -95.578611) # Longitude
    # miny, maxy = min(29.711944, 29.810278, 29.811944, 29.723889, 29.720278, 29.711944), max(29.711944, 29.810278, 29.811944, 29.723889, 29.720278, 29.711944) # Latitude

    # downtown and north Houston
    # minx, maxx = min(-95.410000, -95.411111, -95.281111, -95.277222, -95.410000), max(-95.410000, -95.411111, -95.281111, -95.277222, -95.410000) # Longitude
    # miny, maxy = min(29.745833, 29.849444, 29.851389, 29.746667, 29.745833), max(29.745833, 29.849444, 29.851389, 29.746667, 29.745833) # Latitude

    # pat = pat[(pat.longitude > minx) & (pat.longitude < maxx) & (pat.latitude > miny) & (pat.latitude < maxy)]



    
    #------------------------
    ## analyze patterns data
    
    # subset sectors
    pat["naics_code"] = pat["naics_code"].astype(str)
    # produce a count of 2-digit naics sectors
    pat.groupby(pat.naics_code.str[:2]).count()
    #naics_list = ['51', '52', '54', '55', '56', '81', '92'] #offices
    naics_list = ['44', '45'] # retail
    #naics_list = ['72'] # accommodation and food services
    #pat = pat[pat['naics_code'].str[:2].isin(naics_list)]

    # convert spend_by_day to list
    pat = pat[~pat['visits_by_day'].isnull()]
    pat["list"] = pat["visits_by_day"].apply(lambda x: ast.literal_eval(x))
    
    # Explode the list
    pat = pat.explode("list")
    pat['position'] = pat.groupby(pat.index).cumcount()

    # create date column
    pat["date"] = pd.to_datetime(pat["date_range_start"], utc=True)
    pat["date"] = pat["date"] + pd.to_timedelta(pat["position"], unit="d")
    pat['day'] = pat['date'].dt.date
    
    #pat = pat[pat.top_category == 'Restaurants and Other Eating Places']
    # 44-45 Retail Trade
    

    
    
    # # accommodation and food services
    # pat = pat[pat['naics_code'].str[:2] == '72']
    
    # get days of extreme event
    powday = "2022-06-15"
    endday = "2022-06-18"
    dayone = pat[pat.day.between(pd.to_datetime(powday,utc=True), pd.to_datetime(endday,utc=True))]
    
    dayone = dayone[['placekey', 'latitude', 'longitude', 'list']]
    dayone = dayone.groupby(["placekey", "latitude", "longitude"]).mean().reset_index()
    
    #dayone = pat[pat.day == pd.to_datetime(powday)]
    
    # get same day of the week in the two weeks before from spend dataframe and assign to separate pandas df
    # create new column that contains the day of the week (1=Sunday, 2=Monday, ..., 7=Saturday)
    pat['day_of_week'] = pat['date'].dt.dayofweek
    
    start = "2022-05-23"
    end = "2022-05-29"
    datelist = [datetime.strptime(date, "%Y-%m-%d").date() for date in ["2022-06-07", "2022-05-24", "2022-05-17", "2022-05-10"]]
    before = pat[(pat.date <= pd.to_datetime(end, utc=True)) & (pat.date >= pd.to_datetime(start, utc=True))]
    #before = pat[pat.day.isin(datelist)]
    #before = pat[pat.date < pd.to_datetime("2021-02-10", utc=True)]
    before = before[['placekey', 'list']]

    # get the average spend value
    average_spend = before.groupby("placekey").mean().reset_index()
    average_spend.rename({"list": "spend_average"}, axis=1, inplace=True)
    
    # divide dayone by average_spend
    dayone = dayone.merge(average_spend, on="placekey")
    dayone = dayone[dayone["spend_average"] > 2]
    dayone["spend_norm"] = dayone["list"]  / dayone["spend_average"]
    
    dayone['spend_norm'] = dayone['spend_norm'].astype(float)
    
    
    
    #------------------------
    ## analyze spend data
    
    spend_dir = "/Users/davidvandijcke/Dropbox (University of Michigan)/dynamic/repo/data/spend/2022/20220601-safegraph_sp_spend_patterns_0.csv.gz"

    spend = pd.read_csv(spend_dir)

    
    # filter spend data on places using an anti-join
    spend = spend.merge(places, on="placekey", how="inner")
    
    # convert spend_by_day to list
    spend["list"] = spend["spend_by_day"].apply(lambda x: ast.literal_eval(x))
    
    # Explode the list
    spend = spend.explode("list")
    spend['position'] = spend.groupby(spend.index).cumcount()

    # create date column
    spend["date"] = pd.to_datetime(spend["spend_date_range_start"], utc=True)
    spend["date"] = spend["date"] + pd.to_timedelta(spend["position"], unit="d")
    spend['day'] = spend['date'].dt.date
    
    

    # get day of power outage
    dayone = spend[spend.day.between(pd.to_datetime(powday), pd.to_datetime(endday))]
    
    dayone = dayone[['placekey', 'latitude', 'longitude', 'list']]
    dayone = dayone.groupby(["placekey", "latitude", "longitude"]).mean().reset_index()
    
    # get same day of the week in the two weeks before from spend dataframe and assign to separate pandas df
    # create new column that contains the day of the week (1=Sunday, 2=Monday, ..., 7=Saturday)
    spend['day_of_week'] = spend['date'].dt.dayofweek
    
    
    #before = spend[(spend.date < powday) & (spend.date >= pd.to_datetime(powday, utc=True) - pd.to_timedelta(14, unit="d")) & (spend.day_of_week == 4)]
    before = spend[spend.day == pd.to_datetime(start, utc=True)]
    before = before[['placekey', 'list']]

    # get the average spend value
    average_spend = before.groupby("placekey").mean().reset_index()
    average_spend.rename({"list": "spend_average"}, axis=1, inplace=True)
    
    # divide dayone by average_spend
    dayone = dayone.merge(average_spend, on="placekey")
    dayone = dayone[dayone["spend_average"] > 1]
    dayone["spend_norm"] = dayone["list"] # / dayone["spend_average"]
    
    dayone['spend_norm'] = dayone['spend_norm'].astype(float)
    
    # plot spend_norm on map
    # create geodataframe
    gdf = gpd.GeoDataFrame(dayone['spend_norm'], geometry=gpd.points_from_xy(dayone.longitude, dayone.latitude))
    gdf.crs = "EPSG:4326"
    
    # bounding box around Downtown Houston
    # minx, maxx = -95.383064, -95.353519 # Longitude
    # miny, maxy = 29.745942, 29.767675 # Latitude

    # filter on bounding box
    #gdf = gdf.cx[minx:maxx, miny:maxy]
    
    #gdf.loc[gdf.spend_norm > 5, 'spend_norm'] = 5
    gdf = gdf.to_crs(epsg=3857)
    
    #gdf = gdf[gdf.spend_norm == 0]
    
    # plot
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, markersize=10, alpha=0.5, cmap='RdYlBu_r', column='spend_norm', legend=True, vmin=0,vmax=2)
    ctx.add_basemap(ax)
    plt.show()
    
    # dayone trim outliers
    temp = dayone[~dayone.spend_norm.isna()].copy()
    temp.loc[temp.spend_norm > 2, 'spend_norm'] = 2
    
    # preprocess for segmentation
    X = np.stack([np.array(temp['latitude']), np.array(temp['longitude'])], axis = 1)
    Y = np.array(temp['spend_norm'])
    
    # cast data to raster
    
    plt.scatter(X[:,1], X[:,0], c=Y, cmap = 'RdYlBu_r')
    
    resolution = 1/int(np.sqrt(0.5*Y.size))
    

    model = FDD(Y=Y, X = X, level = 16, lmbda = 5, nu = 0.001, iter = 10000, tol = 5e-5, 
        pick_nu = "MS", scaled = True, scripted = False, image=False, rectangle=True, resolution=resolution)
    
    plt.imshow(model.grid_y, cmap = 'RdYlBu_r', origin="lower")
    plt.colorbar()
    
    # # bounding box around Berkeley, California
    # minx, maxx = -122.324920, -122.234720 # Longitude
    # miny, maxy = 37.835434, 37.905950 # Latitude
    
    # bounding box around Houston, Texas
    minx, maxx = -95.823268, -95.069705 # Longitude
    miny, maxy = 29.523105, 30.110731 # Latitude

    
    
    
    # plot bounding box
    bbox = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326")
    gdf = gdf.to_crs(epsg=3857)
    ax = gdf.plot(alpha=0, edgecolor='k')
    ctx.add_basemap(ax)

    # filter dayone on bounding box
    dayone = dayone[(dayone["longitude"] > minx) & (dayone["longitude"] < maxx) & (dayone["latitude"] > miny) & (dayone["latitude"] < maxy)]
    
    # convert to geo
    
    







# Define grid size
grid_size = (30, 30) # e.g. 100x100 grid

# Define grid boundaries (you might want to adjust these)
lat_min, lat_max = min(X[:,1]), max(X[:,1])
lon_min, lon_max = min(X[:,0]), max(X[:,0])

# Define the grid edges
lat_edges = np.linspace(lat_min, lat_max, grid_size[0]+1)
lon_edges = np.linspace(lon_min, lon_max, grid_size[1]+1)

# Use scipy's binned_statistic_2d function to bin the data
statistic, lat_bin, lon_bin, bin_number = binned_statistic_2d(
    x = X[:,1], 
    y = X[:,0], 
    values = Y, 
    statistic = 'mean', # You can change this to 'median', 'sum', etc. depending on your needs
    bins = [lat_edges, lon_edges]
)

# Plotting
plt.figure(figsize=(10,10))
plt.imshow(statistic.T,  extent=[lat_min, lat_max, lon_min, lon_max], origin = 'upper', cmap='RdYlBu_r')
plt.colorbar(label='Mean spend_norm')
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.show()