import pandas as pd 
import os
os.environ['USE_PYGEOS'] = '0'
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import box, Polygon, Point
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
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

from uhi import *

if __name__ == "main":
    
    