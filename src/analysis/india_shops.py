## TODO: try aggregating to districts

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
import requests
import json


import ast
import matplotlib as mpl

def moveUp(fn, times = 1):
    for _ in range(times):
        fn = os.path.dirname(fn)
    return fn

def get_shops(bbox):
    overpass_url = "http://overpass-api.de/api/interpreter"

    # overpass_query = f"""
    # [out:json][timeout:200];
    # (
    # node[amenity=restaurant]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node[amenity=cafe]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node[amenity=fast_food]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node[amenity=bar]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node[amenity=pub]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node[amenity=biergarten]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node[amenity=food_court]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node[amenity=ice_cream]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node["building"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node["building"="kiosk"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node["building"="supermarket"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node["building"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node["landuse"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node["landuse"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node["office"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node["industrial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # node["shops"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way["building"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way["building"="kiosk"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way["building"="supermarket"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way["building"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way["landuse"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way["landuse"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way["office"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way["industrial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # relation["building"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # relation["building"="kiosk"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # relation["building"="supermarket"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # relation["building"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # relation["landuse"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # relation["landuse"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # relation["office"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # );
    # out body;
    # >;
    # out skel qt;
    # out geom;
    # """
    
    #     way[amenity]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[shop]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[office]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[industrial]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[craft]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[landuse=retail]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[landuse=commercial]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[landuse=industrial]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[amenity=bus_station]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[amenity=taxi]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[amenity=railway_station]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[amenity=aerodrome]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[amenity=parking]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[amenity=hotel]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    # way[amenity=guest_house]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    
    overpass_query = f"""
    [out:json][timeout:300];
    (
    node[amenity]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[shop]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[office]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[industrial]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[craft]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[landuse=retail]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[landuse=commercial]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[landuse=industrial]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[amenity=bus_station]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[amenity=taxi]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[amenity=railway_station]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[amenity=aerodrome]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[amenity=parking]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[amenity=hotel]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[amenity=guest_house]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    node[amenity=zoo]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
    );
    out body;
    >;
    out skel qt;
    out geom;
    """
    response = requests.get(overpass_url, 
                            params={'data': overpass_query})
    data = response.json()
    
    

    # To check the result
    print(json.dumps(data, indent=4))
    
    # convert to dataframe
    shops = pd.DataFrame(data["elements"])
    
    # shops.loc[shops['geometry'].apply(type) == list, 'coordinates'] = shops.loc[shops['geometry'].apply(type) == list, 'geometry'].apply(lambda x: [(coord['lon'], coord['lat']) for coord in x])

    # # Create a Polygon from the coordinates and set it as the geometry
    # shops['geometry'] = np.nan
    # shops.loc[~shops.coordinates.isna(), 'geometry'] = shops.loc[~shops.coordinates.isna(), 'coordinates'].apply(lambda x: Polygon(x) if len(x) > 2 else np.nan)
    # shops.loc[~shops.coordinates.isna(), 'geometry'] = shops.loc[~shops.coordinates.isna(),'coordinates'].apply(lambda x: Polygon(x))
    # shops.loc[shops.coordinates.isna(), 'geometry'] = gpd.points_from_xy(shops.loc[shops.coordinates.isna(), 'lon'], shops.loc[shops.coordinates.isna(), 'geometry'])
    
    shops['geometry'] = gpd.points_from_xy(shops['lon'], shops['lat'])

    
    shops = gpd.GeoDataFrame(shops, geometry=shops.geometry, crs="epsg:4326")
    shops = shops.to_crs("epsg:3857")
    
    return shops

def cast_to_grid_shops(df, shops):
    
    
    df = df.to_crs(epsg=3857)  # Project into Mercator (units in meters)

    # Get bounds
    minx, miny, maxx, maxy = df.geometry.total_bounds

    # merge
    df = gpd.sjoin(shops, df, how="left", op="intersects")
    
    # aggregate
    df = df.groupby("geometry").agg({"index_right": "count"}).reset_index()

    return df

def cast_to_grid(df, meters=5000, agg="sum", op = "within"):
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
    count = gpd.sjoin(df, grid_gdf, how="inner", op=op).groupby("index_right").agg({"pings": agg})

    # join the counts back to the grid
    grid_gdf = grid_gdf.join(count, how="left")
    grid_gdf["pings"] = grid_gdf["pings"].fillna(0)
    
    return grid_gdf

def cast_to_admin(grid_gdf, gadm, gid="GID_3"):
    grid_gdf = gpd.sjoin(grid_gdf, gadm, how="left", op="intersects")
    grid_gdf = grid_gdf.groupby(gid).agg({"pings": "sum"}).reset_index()
    grid_gdf = gadm.merge(grid_gdf, on=gid, how="left")
    
    return grid_gdf



if __name__ == '__main__':
    
    #spend_list = [spend_dir + "m=" + str(x) + "/" for x in [9,10]]
    
    dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir)

    # get directory above
    main_dir = moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    

    #---
    # parameters
    fn_india = os.path.join(data_in, 'india', 'rajasatan_sundays_sep_2021')
    cheatdate =  "2021-09-26"
    gid = "GID_3"
    gidname = "gadm41_IND_3.json"
    meters = 40000
    #---
    
    fn = os.path.join(data_out, 'india', 'rajasatan_cheating')
    
    # read csv file inside fn folder
    files_in_directory = os.listdir(fn)
    csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]

    cheat = pd.read_csv(*csv_files, compression="gzip")
    
    fn = os.path.join(data_out, 'india', 'rajasatan_cheating_before')
    
    # read csv file inside fn folder
    files_in_directory = os.listdir(fn)
    csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]
    before = pd.read_csv(*csv_files, compression="gzip")

    
    # convert to geopandas
    gdf = gpd.GeoDataFrame(cheat, geometry=gpd.points_from_xy(cheat.longitude, cheat.latitude))
    gdf = gdf.set_crs("epsg:4326")
    bbox = list(gdf.total_bounds)
    gdf = gdf.to_crs(epsg=3857)  # Project into Mercator (units in meters)

    
    before = gpd.GeoDataFrame(before, geometry=gpd.points_from_xy(before.longitude, before.latitude))
    before = before.set_crs("epsg:4326")
    before = before.to_crs(epsg=3857)  # Project into Mercator (units in meters)

    redo_shops = False
    fn = os.path.join(data_out, 'india', 'shops.shp')

    if redo_shops:
        shops = get_shops(bbox)
        shops['geometry'] = shops.buffer(150) # 150 meter buffer around shop points
        shops.to_file(fn)
    
    shops = gpd.read_file(fn)
        #---- overlay grid onto data

    #------ 
    # day of
    if redo_shops:
        grid_gdf = cast_to_grid_shops(gdf, shops)
        grid_gdf.rename(columns={"index_right":'pings'}, inplace=True)
        grid_gdf = gpd.GeoDataFrame(grid_gdf, geometry=grid_gdf.geometry)
        grid_gdf.to_file(os.path.join(data_out, 'india', 'rajasatan_cheating_grid_shops.geojson'), driver='GeoJSON')
        
    # reload day of
    grid_gdf = gpd.read_file(os.path.join(data_out, 'india', 'rajasatan_cheating_grid_shops.geojson'))
    
    # cast to admin regions
    gadm = gpd.read_file(os.path.join(data_in, 'india', gidname)).to_crs("epsg:3857")
    # grid_gdf = cast_to_admin(grid_gdf, gadm, gid=gid)
    # grid_gdf = grid_gdf[~grid_gdf.pings.isna() ]
    
    grid_gdf = cast_to_grid(grid_gdf, meters=meters, agg="sum", op="intersects")
    
    
    # read csv file inside fn folder
    fn = os.path.join(data_out, 'india', 'rajasatan_cheating')
    files_in_directory = os.listdir(fn)
    csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]

    cheat = pd.read_csv(*csv_files, compression="gzip")
    cheat = gpd.GeoDataFrame(cheat, geometry=gpd.points_from_xy(cheat.longitude, cheat.latitude), crs = "epsg:4326")
    cheat = cheat.to_crs("epsg:3857")
    cheat['pings'] = 1
    cheat = cast_to_admin(cheat,gadm,gid=gid)
    cheat = cheat[~cheat['pings'].isna()]
    cheat.rename(columns = {'pings' : 'count'}, inplace=True)
    
    grid_gdf = cheat.sjoin(grid_gdf, how="right", op="intersects").groupby("geometry").agg({'pings' : 'mean', 'count' : 'mean'}).reset_index()
    # merge back on index_right for cheat and index for grid_gdf
    grid_gdf['count'] = grid_gdf['count'].fillna(0)
    grid_gdf = gpd.GeoDataFrame(grid_gdf, geometry=grid_gdf.geometry)
    
    #---------------
    # days before
    
    if redo_shops:
        fn = os.path.join(data_out, 'india', 'before_temp.csv')
        before.to_csv(fn,  index=False)
        chunk_size = int(before.size/16)  # specify the chunk size
        result = pd.DataFrame()  # create an empty dataframe to hold results

        for chunk in pd.read_csv(fn, chunksize=chunk_size):  
            chunkgdf = gpd.GeoDataFrame(chunk.drop(columns="geometry"), geometry=gpd.points_from_xy(chunk.longitude, chunk.latitude), crs="epsg:4326")
            temp_result = cast_to_grid_shops(chunkgdf, shops)
            result = pd.concat([result, temp_result])
            
        result.rename(columns={"index_right":'pings'}, inplace=True)
        grid_before = result.groupby("geometry").sum().reset_index()  # store the final result
        
        grid_before = gpd.GeoDataFrame(grid_before, geometry=grid_before.geometry, crs="epsg:3857")
        grid_before.to_file(os.path.join(data_out, 'india', 'rajasatan_cheating_grid_before_shops.geojson'), driver='GeoJSON')
        
    # reload before
    grid_before = gpd.read_file(os.path.join(data_out, 'india', 'rajasatan_cheating_grid_before_shops.geojson'))
    grid_before['pings'] = grid_before['pings'] / 3
    #grid_before = cast_to_admin(grid_before, gadm, gid=gid)
    # grid_before = grid_before[~grid_before.pings.isna() ]

    # left off here
    grid_before.rename(columns = {'pings' : 'pings_before'}, inplace=True)
    test = (grid_gdf.sjoin(grid_before, how="left", op="intersects")
                    .groupby("geometry").agg({'pings' : 'mean', 'pings_before' : 'sum', 'count' : 'mean'}).reset_index())
    test['pings_before'] = test['pings_before'].fillna(0)
    grid_gdf = test.copy()
    grid_gdf = gpd.GeoDataFrame(grid_gdf, geometry=grid_gdf.geometry)
    
    fn = os.path.join(data_out, 'india', 'rajasatan_cheating_before')
    files_in_directory = os.listdir(fn)
    csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]

    cheat = pd.read_csv(*csv_files, compression="gzip")
    cheat = gpd.GeoDataFrame(cheat, geometry=gpd.points_from_xy(cheat.longitude, cheat.latitude), crs = "epsg:4326")
    cheat = cheat.to_crs("epsg:3857")
    #cheat['pings'] = 1
    # cheat = cast_to_admin(cheat,gadm,gid=gid)
    # cheat = cheat[~cheat['pings'].isna()]
    #cheat.rename(columns = {'pings' : 'count_before'}, inplace=True)
    
    grid_gdf = cheat.sjoin(grid_gdf, how="right", op="intersects").groupby("geometry").agg({'pings' : 'mean', 'count' : 'mean', 
                                                                          'pings_before' : 'mean', 'caid' : 'count'}).reset_index()
    grid_gdf.rename(columns = {'caid' : 'count_before'}, inplace=True)

    # merge back on index_right for cheat and index for grid_gdf
    grid_gdf['count_before'] = grid_gdf['count'].fillna(0)
    grid_gdf = gpd.GeoDataFrame(grid_gdf, geometry=grid_gdf.geometry)
    
    merged = grid_gdf.copy()        
    
    merged["pings_norm"] = merged["pings"] / merged["pings_before"]
    merged['pings_share'] = merged['pings'] / merged['count']
    merged.loc[(merged['pings'] == 0) | (merged['pings_before'] == 0), 'pings_norm'] = 0
    merged.loc[(merged['pings'] == 0) | (merged['count'] == 0), 'pings_share'] = 0
    merged['pings_share_pre'] = merged['pings_before'] / (merged['count_before']/3)
    merged.loc[(merged['pings_before'] == 0) | (merged['count_before'] == 0), 'pings_share_pre'] = 0
    merged['pings_share_change'] = merged['pings_share'] - merged['pings_share_pre']
    merged['pings_share_norm'] = merged['pings_share_change'] / merged['pings_share_pre']
    merged.loc[merged.pings_share_pre == 0, 'pings_share_norm' ] = 0
    
    merged = gpd.GeoDataFrame(merged, geometry=merged.geometry)
    
    merged.to_file(os.path.join(data_out, 'india', 'rajasatan_cheating_shops_merged_40K.geojson'), driver='GeoJSON')
    # calculate total pings
    
    
    #---------------------
    merged = gpd.read_file(os.path.join(data_out, 'india', 'rajasatan_cheating_shops_merged_40K.geojson'), driver='GeoJSON')

    
    
    
    
    
    gdf = merged.copy()
    #### plot figure
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

    # Create the plot
    fig, ax = plt.subplots(1, 1)

    # Plot the data
    vmax = 1.5
    p = gdf.plot(column="pings_norm", cmap='coolwarm',  ax=ax, legend=True, vmax=vmax)

    # Plot the Rajasthan border
    rajasthan_geometry.boundary.plot(color='k', linewidth=2, ax=ax, alpha=0.5)

    # Remove padding and axis
    ax.set_axis_off()

    # Set the plot limits to match the defined zoom area
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("equal")
    
    
    
        # Create a custom colorbar
    norm = mpl.colors.Normalize()
    cbar = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
    fig.colorbar(cbar, ax=ax, orientation="vertical", pad=0, label=f"% of Monthly Average")
