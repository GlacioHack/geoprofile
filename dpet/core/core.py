import geoviews as gv
import holoviews as hv
from holoviews.streams import PointDraw
import math
import numpy as np
import pandas as pd
import panel as pn
import pathlib
import rasterio

import dpet

hv.extension('bokeh')

def dense_points_along_transect(list_of_coordinate_tuples, num_points_between_coords=100):
    """
    Creates dense list of coordinates along transect
    
    Parameters
    ----------
    list_of_coordinate_tuples : list([tuple(x,y), tuple(x,y),...])
    
    num_points_between_coords : int
                  
    Returns
    -------
    x_coordinates : list([tuple(x,y), tuple(x,y),...])
    y_coordinates : list([tuple(x,y), tuple(x,y),...])
    """
    
    x_coordinates = []
    y_coordinates = []
    
    c=0

    for coordinate in list_of_coordinate_tuples:
        if c < len(list_of_coordinate_tuples)-1:
            start_x = coordinate[0]
            stop_x  = list_of_coordinate_tuples[c+1][0]
            start_y = coordinate[1]
            stop_y  = list_of_coordinate_tuples[c+1][1]

            x_added_coordinates = np.linspace(start_x, stop_x, num=num_points_between_coords)
            y_added_coordinates = np.linspace(start_y, stop_y, num=num_points_between_coords)

            x_coordinates.extend(x_added_coordinates)
            y_coordinates.extend(y_added_coordinates)

            c+=1
    
    return x_coordinates, y_coordinates

def distance_along_transect(list_of_coordinate_tuples):
    
    cumulative_distance_along_transect = 0
    distances = [0]
    c=0
    for coordinate in list_of_coordinate_tuples:
        if c < len(list_of_coordinate_tuples)-1:
            x1 = coordinate[0]
            y1 = coordinate[1]
            x2 = list_of_coordinate_tuples[c+1][0]
            y2 = list_of_coordinate_tuples[c+1][1]
            
            distance = distance_between_two_points(x1,y1,x2,y2)
            cumulative_distance_along_transect += distance
            
            distances.append(cumulative_distance_along_transect)

            c+=1
    
    return distances

def distance_between_two_points(x1,y1,x2,y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance
    
def extract_profiles(list_of_coordinate_tuples, 
                     dem_file_list):
                     
    """
    Extracts profiles from multiple DEMs.
    
    Parameters
    ----------
    list_of_coordinate_tuples : list([tuple(x,y), tuple(x,y),...])
    
    dem_file_list : list
                     
    no_data_value : 
                  
    Returns
    -------
    dense_coordinates : list([tuple(x,y), tuple(x,y),...])
    """
    
    x_coordinates, y_coordinates = dpet.core.dense_points_along_transect(list_of_coordinate_tuples)
    dense_coordinates = list(zip(x_coordinates, y_coordinates))
    distances = dpet.core.distance_along_transect(dense_coordinates)
    
    df = pd.DataFrame.from_dict({'distance': distances,
                                 'x_coordinates': x_coordinates,
                                 'y_coordinates': y_coordinates})
    
    for dem_file in dem_file_list:
        base_name = pathlib.Path(dem_file).stem +'_elev_m'
        elevations, no_data_value = dpet.core.sample_dem(dem_file, dense_coordinates)
        df[base_name] = elevations
        df.loc[df[base_name]==no_data_value,base_name] = np.nan # replace no data values with nan
        
    return df
    
def plot_base_map(bounds,
                  dem_crs,
                  points,
                  max_points=5,
                  url= 'https://mt1.google.com/vt/lyrs=s&x={X}&y={Y}&z={Z}'):
    
    """
    Plots base map with Google Sattelite tiles for interactive point selection along transect.
    
    Parameters
    ----------
    bounds : iterable
        left, bottom, right, top
    
    dem_crs : cartopy._epsg._EPSGProjection
                  
    points : geoviews.element.geo.Points
                  
    Returns
    -------
    base_map : holoviews.core.overlay.Overlay
    """
                  
    tiles = gv.WMTS(url, extents=bounds, crs=dem_crs)
    base_map = (tiles * points).opts(gv.opts.Points(width=900, 
                                                     height=900, 
                                                     size=5, 
                                                     color='blue', 
                                                     tools=["hover"]))
    return base_map
    

def sample_dem(dem_file, list_of_coordinate_tuples):
    source = rasterio.open(dem_file)
    no_data_value = source.nodata
    
    elevations = []
    for val in source.sample(list_of_coordinate_tuples):
        elevations.append(val[0])
    
    return elevations, no_data_value