import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Define the path to your file
file_path = "/path/to/your/file.h5"

# Define the bounding box coordinates
min_lon, max_lon = 70, 80
min_lat, max_lat = 20, 30

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

# Define a cap value - let's take the 99th percentile as the cap
cap_value = np.nanpercentile(data_scaled, 99)

# Cap the values
data_scaled[data_scaled > cap_value] = cap_value

# Create a figure
fig, ax = plt.subplots(figsize=(10, 10))

# Define the map
m = Basemap(projection='cyl', resolution='l', 
            llcrnrlat=min_lat, urcrnrlat=max_lat,
            llcrnrlon=min_lon, urcrnrlon=max_lon, ax=ax)

# Draw coastlines and countries
m.drawcoastlines()
m.drawcountries()

# Draw parallels and meridians.
parallels = np.arange(-90., 91., 10.)
m.drawparallels(parallels, labels=[True, False, False, False])
meridians = np.arange(-180., 181., 10.)
m.drawmeridians(meridians, labels=[False, False, False, True])

# Plot the scaled data
img = m.imshow(data_scaled, origin='upper', cmap='gray')

# Add a colorbar
plt.colorbar(img, ax=ax, orientation='vertical', label='Scaled Radiance (nWatts/(cm^2 sr))')

# Set the title
plt.title('Scaled Gap Filled BRDF Corrected DNB Radiance')

plt.show()
