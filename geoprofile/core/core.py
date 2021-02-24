import geoviews as gv
import holoviews as hv
from holoviews.operation.datashader import regrid
from holoviews.streams import PointDraw
import math
import numpy as np
import pandas as pd
import panel as pn
import pathlib
import rasterio
import rioxarray
import xarray as xr

import geoprofile

hv.extension("bokeh")


def calculate_hillshade(array,
                        azimuth=315,
                        angle_altitude=45):
    """
    Compute hillshaded relief from Digital Elevation Model data array.

    Parameters
    ----------
    array : numpy.ndarray

    azimuth : int
    
    angle_altitude : int

    Returns
    -------
    hillshade : numpy.ndarray
    """
    
    azimuth = 360.0 - azimuth 

    x, y = np.gradient(array)
    slope = np.pi/2.-np.arctan(np.sqrt(x*x+y*y))
    aspect = np.arctan2(-x,y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad) * \
             np.sin(slope) + \
             np.cos(altituderad) * \
             np.cos(slope) * \
             np.cos((azimuthrad-np.pi/2.) - \
             aspect)
    
    hillshade = 255*(shaded + 1)/2

    return hillshade


def dense_points_along_transect(
    list_of_coordinate_tuples, num_points_between_coords=100
):
    """
    Creates dense list of coordinates along transect.

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

    c = 0

    for coordinate in list_of_coordinate_tuples:
        if c < len(list_of_coordinate_tuples) - 1:
            start_x = coordinate[0]
            stop_x = list_of_coordinate_tuples[c + 1][0]
            start_y = coordinate[1]
            stop_y = list_of_coordinate_tuples[c + 1][1]

            x_added_coordinates = np.linspace(
                start_x, stop_x, num=num_points_between_coords
            )
            y_added_coordinates = np.linspace(
                start_y, stop_y, num=num_points_between_coords
            )

            x_coordinates.extend(x_added_coordinates)
            y_coordinates.extend(y_added_coordinates)

            c += 1

    return x_coordinates, y_coordinates


def distance_along_transect(list_of_coordinate_tuples):

    cumulative_distance_along_transect = 0
    distances = [0]
    c = 0
    for coordinate in list_of_coordinate_tuples:
        if c < len(list_of_coordinate_tuples) - 1:
            x1 = coordinate[0]
            y1 = coordinate[1]
            x2 = list_of_coordinate_tuples[c + 1][0]
            y2 = list_of_coordinate_tuples[c + 1][1]

            distance = distance_between_two_points(x1, y1, x2, y2)
            cumulative_distance_along_transect += distance

            distances.append(cumulative_distance_along_transect)

            c += 1

    return distances


def distance_between_two_points(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


def extract_profiles(list_of_coordinate_tuples, dem_file_list):

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

    x_coordinates, y_coordinates = geoprofile.core.dense_points_along_transect(
        list_of_coordinate_tuples
    )
    dense_coordinates = list(zip(x_coordinates, y_coordinates))
    distances = geoprofile.core.distance_along_transect(dense_coordinates)

    df = pd.DataFrame.from_dict(
        {
            "distance": distances,
            "x_coordinates": x_coordinates,
            "y_coordinates": y_coordinates,
        }
    )

    for dem_file in dem_file_list:
        base_name = pathlib.Path(dem_file).stem + "_elev_m"
        elevations, no_data_value = geoprofile.core.sample_dem(dem_file, dense_coordinates)
        df[base_name] = elevations
        df.loc[
            df[base_name] == no_data_value, base_name
        ] = np.nan  # replace no data values with nan

    return df


def plot_base_map_tiles_gui(
    bounds, dem_crs, points, url="https://mt1.google.com/vt/lyrs=s&x={X}&y={Y}&z={Z}"
):

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
    base_map = (tiles * points).opts(
        gv.opts.Points(width=900, height=900, size=5, color="blue", tools=["hover"])
    )
    return base_map


def plot_base_map_dem_gui(dem_file, points):

    """
    Plots dem for interactive point selection along transect.

    Parameters
    ----------
    dem_file : path to GeoTIFF file on disk.

    points : geoviews.element.geo.Points

    Returns
    -------
    base_map : holoviews.core.overlay.Overlay
    """

    dem_gv = (
        gv.Dataset(gv.load_tiff(dem_file, nan_nodata=True))
        .to(gv.Image, ["x", "y"])
        .opts(tools=["hover"])
    )
    dem_gv = regrid(dem_gv.opts(colorbar=True, cmap="greys"))

    base_map = (dem_gv * points).opts(
        gv.opts.Points(width=750, height=900, size=5, color="blue", tools=["hover"])
    )
    return base_map

def replace_and_fill_nodata_value(array, nodata_value, fill_value):
    """
    Replace nodata values with fill value in array.
    
    Parameters
    ----------
    array : numpy.ndarray

    nodata_value : value similar to array.dtype
    
    fill_value : value similar to array.dtype

    Returns
    -------
    masked_array : numpy.ndarray
    """
    
    mask = (array == nodata_value)
    masked_array = np.ma.masked_array(array, mask=mask)
    masked_array = np.ma.filled(masked_array, fill_value=fill_value)
    
    return masked_array

def sample_dem(dem_file, list_of_coordinate_tuples):
    source = rasterio.open(dem_file)
    no_data_value = source.nodata

    elevations = []
    for val in source.sample(list_of_coordinate_tuples):
        elevations.append(val[0])

    return elevations, no_data_value

def xr_read_GeoTIFF(GeoTIFF_file_path, masked=True):
    """
    Reads in single or multi-band GeoTIFF as chunked dask array for lazy io.
    
    Parameters
    ----------
    GeoTIFF_file_path : str

    Returns
    -------
    ds : xarray.Dataset
        Includes rioxarray extension to xarray.Dataset
    """
    
    da = rioxarray.open_rasterio(GeoTIFF_file_path, chunks=1000, masked=masked)
    
    # Extract bands and assign as variables in xr.Dataset()
    ds = xr.Dataset()
    for i,v in enumerate(da.band):
        da_tmp = da.sel(band=v)
        da_tmp.name = 'band'+str(i+1)
        
        ds[da_tmp.name] = da_tmp
        
    # Delete empty band coordinates.
    # Need to preserve spatial_ref coordinate, even though it appears empty.
    # See spatial_ref attributes under ds.coords.variables used by rioxarray extension.
    del ds.coords['band']
    
    # Preserve top-level attributes and extract single value from value iterables e.g. (1,) --> 1
    ds.attrs = da.attrs
    for key, value in ds.attrs.items():
        try:
            if len(value) == 1:
                ds.attrs[key] = value[0]
        except TypeError:
            pass
        
    return ds
