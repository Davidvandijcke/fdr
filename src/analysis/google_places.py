import requests
import pyproj
import itertools
import math
import time

def epsg3587_to_wgs84(x, y):
    """ Convert EPSG:3587 (Web Mercator) coordinates to WGS84. """
    proj_in = pyproj.Proj(init='epsg:3587')
    proj_out = pyproj.Proj(init='epsg:4326')
    lon, lat = pyproj.transform(proj_in, proj_out, x, y)
    return lat, lon

def generate_circle_centers(min_x, min_y, max_x, max_y, radius):
    """ Generate circle centers to cover the bounding box. """
    lat_min, lon_min = epsg3587_to_wgs84(min_x, min_y)
    lat_max, lon_max = epsg3587_to_wgs84(max_x, max_y)
    
    # Calculate distance between points (approximation)
    lat_step = (radius / 111320) * 1.4  # 1 degree latitude ~ 111.32 km, adjusting factor for denser coverage
    lon_step = (radius / (40075000 * math.cos(math.radians(lat_min)) / 360)) * 1.4  # Adjust for longitude

    lat_points = list(drange(lat_min, lat_max, lat_step))
    lon_points = list(drange(lon_min, lon_max, lon_step))

    return itertools.product(lat_points, lon_points)

def drange(start, stop, step):
    """ Range function for decimals. """
    r = start
    while r < stop:
        yield r
        r += step

def get_places(api_key, location, radius, place_type):
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": api_key,
        "location": location,
        "radius": radius,
        "type": place_type
    }
    places = []
    while True:
        response = requests.get(base_url, params=params).json()
        places.extend(response.get('results', []))

        # Check for a next page token
        page_token = response.get('next_page_token')
        if not page_token:
            break

        # Include the next page token in the subsequent request
        params['pagetoken'] = page_token
        time.sleep(2)  # Necessary to wait for the token to become valid

    return places

# Define your bounding box and radius
min_x, min_y, max_x, max_y = [7649981.160830013, 2531547.754779711, 9015022.947024144, 3621930.062059432]
radius = 500  # in meters, adjust as needed

# Generate circle centers
centers = generate_circle_centers(min_x, min_y, max_x, max_y, radius)

# Your API Key
api_key = "YOUR_API_KEY"
place_type = "restaurant"  # Example type

# Collect places from each circle
all_places = []
for lat, lon in centers:
    location = f"{lat},{lon}"
    places = get_places(api_key, location, radius, place_type)
    all_places.extend(places)
    time.sleep(1)  # To respect API rate limits

# Process all_places as needed
