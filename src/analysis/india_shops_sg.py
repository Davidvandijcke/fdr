
import subprocess
import pandas as pd
from shapely import wkt
import geopandas as gpd
import os

def moveUp(fn, times = 1):
    for _ in range(times):
        fn = os.path.dirname(fn)
    return fn

def convert_to_geojson(fout, output_prefix, geom_types=('points', 'lines', 'polygons', 'multipolygons')):
    for geom_type in geom_types:
        output_file = f"{output_prefix}{geom_type}.geojson"
        cmd = f'ogr2ogr -f "GeoJSON" "{output_file}" "{fout}" {geom_type}'
        subprocess.run(cmd, shell=True)
        
def read_multiple_geojsons_to_gdf(filenames):
    # Read each file into a GeoDataFrame and store in a list
    gdfs = [gpd.read_file(filename) for filename in filenames]

    # Concatenate all the GeoDataFrames into a single one
    return pd.concat(gdfs, ignore_index=True)

def getOSM(data_out):
    # get Gaza OSM data
    fp = os.path.join(data_in, 'india', "india-220101.osm.pbf")
    fout = os.path.join(data_in, 'india', "india-220101_shops.pbf")
    cmd = f"osmium tags-filter '{fp}' \
        nwr/shop=* \
        nwr/office=* \
        nwr/amenity=restaurant \
        nwr/amenity=cafe \
        nwr/amenity=bank \
        nwr/amenity=hotel \
        nwr/amenity=guest_house \
        nwr/amenity=bar \
        nwr/amenity=fast_food \
        nwr/amenity=pub \
        nwr/amenity=cinema \
        nwr/amenity=nightclub \
        nwr/amenity=theatre \
        nwr/amenity=marketplace \
        nwr/amenity=car_rental \
        nwr/amenity=car_wash \
        nwr/amenity=car_repair \
        nwr/amenity=bicycle_rental \
        nwr/amenity=taxi \
        nwr/building=retail \
        nwr/building=commercial \
        nwr/landuse=retail \
        nwr/landuse=commercial \
        nwr/landuse=industrial \
        nwr/industrial=* \
        nwr/brand=* \
        -o '{fout}' --overwrite"

    subprocess.run(cmd, shell=True)
    
    
    output_prefix = os.path.join(data_in, 'india', 'osm/')
    convert_to_geojson(fout, output_prefix)


    # Example usage:
    geom_types = ['points', 'multipolygons']

    # Generate the list of filenames
    filenames = [f"{output_prefix}{geom_type}.geojson" for geom_type in geom_types]

    # Read them into a single GeoDataFrame
    gdf = read_multiple_geojsons_to_gdf(filenames)
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    gdf.to_file(os.path.join(data_out, 'osm_buildings_pbf.geojson'), driver='GeoJSON')
    



if __name__ == '__main__':
    
    #spend_list = [spend_dir + "m=" + str(x) + "/" for x in [9,10]]
    
    dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir)

    # get directory above
    main_dir = moveUp(dir, 4)
    data_in = os.path.join(main_dir, 'data', 'in')    
    data_out = os.path.join(main_dir, 'data', 'out')  
    figs_dir = os.path.join(main_dir, 'results', 'figs')

    
    
    
    
    #### Safegraph processing
    
    test = pd.read_csv(os.path.join(data_in, 'safegraph_india', 'safegraph_india.csv'), sep="\t")
    
    temp = test[test.POLYGON_WKT.notna()]
    def _wkt_loads(x):
        try:
            return wkt.loads(x)
        except:
            return None
    temp['POLYGON_WKT'] = temp['POLYGON_WKT'].apply(_wkt_loads)
    temp = temp[temp.POLYGON_WKT != None]
    temp = gpd.GeoDataFrame(temp, geometry=temp['POLYGON_WKT'], crs="epsg:4326")
    
    temp.drop(columns="POLYGON_WKT").to_file(os.path.join(data_out, 'safegraph_india.geojson'), driver='GeoJSON')