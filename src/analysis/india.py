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
    
    #spend_list = [spend_dir + "m=" + str(x) + "/" for x in [9,10]]
    
    dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir)

    # get directory above
    main_dir = moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    
    redo_merge = False
    if redo_merge:
        #---
        # parameters
        fn_india = os.path.join(data_in, 'india', 'rajasatan_sundays_sep_2021')
        cheatdate =  "2021-09-26"
        #---
        
        df = spark.read.parquet(fn_india)
        
        ## day of cheating
        df = df.filter(F.to_date(df.date) == cheatdate) # day of cheating shutdown
        df = df.filter(F.hour(df.date).between(18,24)) # time of cheating shutdown
        
        fn = os.path.join(data_out, 'india', 'rajasatan_cheating_after6')
        df.repartition(1).write.mode("overwrite").csv(fn, header=True, compression="gzip")
        
        # read csv file inside fn folder
        files_in_directory = os.listdir(fn)
        csv_files = [os.path.join(fn, file) for file in files_in_directory if file.endswith(".csv.gz")]

        cheat = pd.read_csv(*csv_files, compression="gzip")
        
        ## days before
        df = spark.read.parquet(fn_india)

        df = df.filter(F.to_date(df.date) != cheatdate) # day before cheating shutdown
        df = df.filter(F.hour(df.date).between(18,24)) # time of cheating shutdown
        
        fn = os.path.join(data_out, 'india', 'rajasatan_cheating_before_after6')
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
        
        fn_merged = os.path.join(data_out, 'india', 'rajasatan_cheating_grid_after6.geojson')
        merged.to_file(fn_merged, driver='GeoJSON')

    else:
        fn_merged = os.path.join(data_out, 'india', 'rajasatan_cheating_grid_after6.geojson')
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

    # Create the plot
    fig, ax = plt.subplots(1, 1)

    # Plot the data
    vmax = 200
    p = gdf.plot(column="count_norm", cmap='coolwarm', vmax=vmax, ax=ax, legend=False)

    # Plot the Rajasthan border
    rajasthan_geometry.boundary.plot(color='k', linewidth=1, ax=ax, alpha=0.5)
    

    # Remove padding and axis
    ax.set_axis_off()

    # Set the plot limits to match the defined zoom area
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("equal")

    # Add a title to the plot

    # Create a custom colorbar
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)
    cbar = plt.cm.ScalarMappable(norm=norm, cmap='coolwarm')
    fig.colorbar(cbar, ax=ax, orientation="vertical", pad=0, label=f"% of Monthly Average")

    plt.savefig(os.path.join(data_out, 'india', 'india_raw.png'), dpi=300, bbox_inches='tight')
    
    
    # make colorbar smaller
    
    
    
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

                # Initialize the OSM object 
        bbox = list(gdf.to_crs("epsg:4326").geometry.total_bounds)
        osm = pyrosm.OSM(fn, bounding_box=bbox)


        # filter on shops
        custom_filter = {"shop": True}
        pois = osm.get_pois(custom_filter=custom_filter)
        
        pois.to_file(fn_json, driver="GeoJSON")

        
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



        # overpass
    
        import requests
        import json

        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = f"""
        [out:json];
        (
        node["building"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["building"="kiosk"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["building"="supermarket"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["building"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["landuse"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["landuse"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["building"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["building"="kiosk"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["building"="supermarket"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["building"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["landuse"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["landuse"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["building"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["building"="kiosk"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["building"="supermarket"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["building"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["landuse"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["landuse"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        );
        out body;
        >;
        out skel qt;
        out geom;
        """
        overpass_query = f"""
        [out:json][timeout:200];
        (
        node[amenity=restaurant]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node[amenity=cafe]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node[amenity=fast_food]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node[amenity=bar]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node[amenity=pub]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node[amenity=biergarten]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node[amenity=food_court]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node[amenity=ice_cream]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["building"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["building"="kiosk"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["building"="supermarket"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["building"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["landuse"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["landuse"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        node["office"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["building"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["building"="kiosk"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["building"="supermarket"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["building"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["landuse"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["landuse"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        way["office"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["building"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["building"="kiosk"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["building"="supermarket"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["building"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["landuse"="commercial"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["landuse"="retail"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
        relation["office"]({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]});
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
        
        shops.to_csv(os.path.join(data_out, "india_shops.csv"), index=False)
        
        # convert geometry to polygon
        shops = pd.read_csv(os.path.join(data_out, "india_shops.csv"))
        # Convert the list of dictionaries to a list of tuples
        shops['coordinates'] = np.nan

        shops.loc[shops['geometry'].apply(type) == list, 'coordinates'] = shops.loc[shops['geometry'].apply(type) == list, 'geometry'].apply(lambda x: [(coord['lon'], coord['lat']) for coord in x])

        # Create a Polygon from the coordinates and set it as the geometry
        shops['geometry'] = np.nan
        shops.loc[~shops.coordinates.isna(), 'geometry'] = shops.loc[~shops.coordinates.isna(), 'coordinates'].apply(lambda x: Polygon(x) if len(x) > 2 else np.nan)
        #shops['geometry'] = shops['coordinates'].apply(lambda x: Polygon(x))
        shops.loc[shops.coordinates.isna(), 'geometry'] = gpd.points_from_xy(shops.loc[shops.coordinates.isna(), 'lon'], shops.loc[shops.coordinates.isna(), 'lat'])
        
        shops = gpd.GeoDataFrame(shops, geometry=shops.geometry, crs="epsg:4326")
        shops = shops.to_crs("epsg:3587")
        fig, ax = plt.subplots()
        shops['geometry'].explore()
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        
    else:
        shops = gpd.read_file(fn_json)
        shops = shops.to_crs("epsg:3587")
        p = shops.plot()
        ctx.add_basemap(p, source=ctx.providers.Stamen.TonerLite)
        
    # plot bbox
    p = gpd.GeoSeries([Polygon([(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[3]), (bbox[2], bbox[1])])])
    p = p.set_crs("epsg:4326")
    p = p.to_crs("epsg:3857")
    go = p.plot(color="none")
    ctx.add_basemap(go, source=ctx.providers.Stamen.TonerLite)
    
    
    import pickle
    fto = "/Users/davidvandijcke/Downloads/1D_SURE.pkl"
    with open(fto, "rb") as f:
        res = pickle.load(f)
        
        
#--------------------------
# Meta
#--------------------------

handle = "2097712012772253_"

# day of
meta_cheat = pd.read_csv(os.path.join(data_in, 'meta_sep_2021', handle + "2021-09-26.csv"))

meta_cheat = meta_cheat[meta_cheat.country == "IN"]

gadm = gpd.read_file(os.path.join(data_in, 'gadm41_IND_2.json'))

gadm.rename({"NAME_2": "gadm2_name"}, axis=1, inplace=True)

meta_cheat = meta_cheat.merge(gadm, on="gadm2_name", how = "inner")

meta_cheat = gpd.GeoDataFrame(meta_cheat, geometry=meta_cheat.geometry)

# previous week
meta_cheat_prev = pd.read_csv(os.path.join(data_in, 'meta_sep_2021', handle + "2021-09-19.csv"))

meta_cheat_prev = meta_cheat_prev[meta_cheat_prev.country == "IN"]

# append "prev" to activity_quantile and activity_percentage
meta_cheat_prev.rename({"activity_quantile": "activity_quantile_prev", "activity_percentage": "activity_percentage_prev"}, axis=1, inplace=True)
meta_cheat_prev = meta_cheat_prev[["gadm2_name", "activity_quantile_prev", "activity_percentage_prev", 'business_vertical']]

# merge into meta_cheat based on gadm2_name
meta_cheat = meta_cheat.merge(meta_cheat_prev, on=["gadm2_name", 'business_vertical'], how = "inner")

meta_cheat['activity_quantile_change'] = (meta_cheat['activity_quantile'] - meta_cheat['activity_quantile_prev']) / meta_cheat['activity_quantile_prev']
meta_cheat['activity_percentage_change'] = (meta_cheat['activity_percentage'] - meta_cheat['activity_percentage_prev']) / meta_cheat['activity_percentage_prev']

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
meta_cheat[meta_cheat['business_vertical'] == 'All'].plot(cmap="coolwarm", column = "activity_percentage", legend=True,ax=ax,vmax=200)

# highlight Rajasthan
meta_cheat[meta_cheat.gadm1_name == "Rajasthan"].dissolve().boundary.plot(color='black', linewidth=2, ax=ax, alpha=0.5)

meta_cheat['business_vertical'].unique()