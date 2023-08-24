import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from skimage.measure import block_reduce
import os
import geopandas as gpd
from descartes import PolygonPatch


def moveUp(fn, times = 1):
    for _ in range(times):
        fn = os.path.dirname(fn)
    return fn

def preprocess_and_regrid(file_path):
    # Open the file
    with h5py.File(file_path, 'r') as f:
        # Access the dataset
        data_set = f['HDFEOS']['GRIDS']['VNP_Grid_DNB']['Data Fields']['Gap_Filled_DNB_BRDF-Corrected_NTL']
        # Read the data into a NumPy array
        data = np.array(data_set)
        # Get the scale factor and fill value from the dataset attributes
        scale_factor = data_set.attrs['scale_factor'][0]
        fill_value = data_set.attrs['_FillValue'][0]

    # Replace the fill values with NaN
    data = data.astype(float)  # Convert data to float type for NaN handling
    data[data == fill_value] = np.nan

    # Apply the scale factor
    data_scaled = data * scale_factor

    # Regrid the data to 10km by 10km blocks by taking the mean of each 10x10 block
    # 'block_reduce' calculates the mean of non-NaN pixels. If all pixels are NaN, the result is NaN.
    
    return data_scaled

dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir)

def preprocess_and_regrid_monthly(file_path):
    # Open the file
    with h5py.File(file_path, 'r') as f:
        # Access the dataset
        data_set = f['HDFEOS']['GRIDS']['VIIRS_Grid_DNB_2d']['Data Fields']['AllAngle_Composite_Snow_Free']
        # Read the data into a NumPy array
        data = np.array(data_set)
        # Get the scale factor and fill value from the dataset attributes
        scale_factor = data_set.attrs['scale_factor'][0]
        fill_value = data_set.attrs['_FillValue'][0]

    # Replace the fill values with NaN
    data = data.astype(float)  # Convert data to float type for NaN handling
    data[data == fill_value] = np.nan

    # Apply the scale factor
    data_scaled = data * scale_factor

    # Regrid the data to 10km by 10km blocks by taking the mean of each 10x10 block
    # 'block_reduce' calculates the mean of non-NaN pixels. If all pixels are NaN, the result is NaN.
    #data_regrid = block_reduce(data_scaled, block_size=(50, 50), func=np.nanmean)
    
    return data_scaled

def print_structure(file, indent=''):
    """
    Prints the structure of HDF5 file.

    Args:
    file: h5py.File
        HDF5 file
    indent: str
        Indentation for pretty print
    """

    if isinstance(file, h5py.Dataset):
        print(indent, 'Dataset:', file.name)
    elif isinstance(file, h5py.Group):
        print(indent, 'Group:', file.name)
        for key in file.keys():
            print_structure(file[key], indent + '  ')



# get directory above
main_dir = moveUp(dir, 4)
data_in = os.path.join(main_dir, 'data', 'in')    
data_out = os.path.join(main_dir, 'data', 'out')  

path = os.path.join(data_in, 'satellite', 'india', 'kashmir')

# Preprocess and regrid the data from the first file
first_file_path = os.path.join(path, 'VNP46A3.A2019213.h25v05.001.2021125210958.h5') # 'VNP46A2.A2021270.h25v06.001.2021285022331.h5') # 'VNP46A2.A2021269.h25v06.001.2021285013842.h5')
data_regrid_1 = preprocess_and_regrid_monthly(first_file_path)

# Preprocess and regrid the data from the second file
second_file_path = os.path.join(path, 'VNP46A3.A2019182.h25v05.001.2021125202730.h5')
data_regrid_2 = preprocess_and_regrid_monthly(second_file_path)

# Open the file
# with h5py.File(second_file_path, 'r') as f:
#     print_structure(f)
    
# # Define the file path
# file_path = os.path.join(path, 'VNP46A3.A2021244.h25v06.001.2021286110234.h5')

# # Preprocess the data from the monthly file
# data_regrid_monthly = preprocess_and_regrid_monthly(file_path)

gran = 1
data_regrid_1 = block_reduce(data_regrid_1, block_size=(gran, gran), func=np.nanmean)
data_regrid_2 = block_reduce(data_regrid_2, block_size=(gran, gran), func=np.nanmean)


# Calculate the ratio of the regridded datasets
# Handle division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    data_ratio = data_regrid_2 #/ data_regrid_2
    # data_ratio[data_regrid_2 == 0] = np.nan
    # data_ratio[(data_regrid_2 == 0 ) & (data_regrid_1 == 0)] = 1
    
# data_ratio = data_regrid_1 / data_regrid_2 # data_regrid_monthly #  data_regrid_monthly # 1



# Define the bounding box coordinates
min_lon, max_lon = 70, 80
min_lat, max_lat = 20, 30

# Define the grid lines
parallels = np.arange(-90., 91., 10.)
meridians = np.arange(-180., 181., 10.)

# Plot the ratio without adjusting the color scale
fig, ax = plt.subplots(figsize=(10, 10))
m = Basemap(projection='cyl', resolution='l', 
            llcrnrlat=min_lat, urcrnrlat=max_lat,
            llcrnrlon=min_lon, urcrnrlon=max_lon, ax=ax)
m.drawcoastlines()
m.drawcountries()
m.drawparallels(parallels, labels=[True, False, False, False])
m.drawmeridians(meridians, labels=[False, False, False, True])




# plot state outline 
states = gpd.read_file(os.path.join(data_in, 'india', 'india_states_shapefile'))
kashmir = states[states['name_1'] == 'Jammu and Kashmir']

# Iterate over the geometries in the GeoDataFrame
for geometry in kashmir['geometry']:
    # Extract x and y coordinates from the polygon's exterior
    x, y = geometry.exterior.coords.xy
    # Convert lat/lon to map projection coordinates
    x, y = m(x, y)
    # Plot the outline of Kashmir
    m.plot(x, y, color='red')



img = m.imshow(data_ratio, origin='upper', cmap='viridis', vmax=2, alpha=0.5)




plt.colorbar(img, ax=ax, orientation='vertical', label='Radiance Ratio')
plt.title('Radiance Ratio of Gap Filled BRDF Corrected DNB Radiance')
plt.show()
