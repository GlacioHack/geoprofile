import cartopy
import DemUtils as du
import functools
import geoviews as gv
import geoutils as gu
import holoviews as hv
from holoviews.operation.datashader import regrid
from holoviews.streams import PointDraw
import math
import numpy as np
import operator
import pandas as pd
import panel as pn
import pathlib
import rasterio
from rasterio.warp import Resampling
import rioxarray
import xarray as xr

import geoprofile

hv.extension("bokeh")


def calculate_hillshade(array, azimuth=315, angle_altitude=45):
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
    slope = np.pi / 2.0 - np.arctan(np.sqrt(x * x + y * y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth * np.pi / 180.0
    altituderad = angle_altitude * np.pi / 180.0

    shaded = np.sin(altituderad) * np.sin(slope) + np.cos(altituderad) * np.cos(
        slope
    ) * np.cos((azimuthrad - np.pi / 2.0) - aspect)

    hillshade = 255 * (shaded + 1) / 2

    return hillshade


def dem_time_series_stack(
    dem_files,
    times,
    replace_fill_value_with_nan=True,
    compute_hillshade=True,
    min_bounds=False,
    max_bounds=False,
    min_res=False,
    max_res=False,
    resampling="nearest",
    save_to_nc=False,
    nc_out_dir=None,
):

    # TODO fill out the doc string.
    """
    Resample DEMs in memory to common bounds and resolution, then stack as time
    dimensionsed xarray.Dataset().

    Parameters
    ----------
    dem_files : list of file paths

    Returns
    -------
    ds : xr.Dataset()
    """

    if isinstance(resampling, type(Resampling.nearest)):
        resampling = resampling
    elif resampling == "nearest":
        resampling = Resampling.nearest
    elif resampling == "cubic":
        resampling = Resampling.cubic
    else:
        resampling = Resampling.nearest

    min_max_bounds_res = geoprofile.core.get_min_max_bounds_res(dem_files)
    min_bounds_val, max_bounds_val, min_res_val, max_res_val = min_max_bounds_res

    if min_bounds:
        bounds = min_bounds_val
    elif max_bounds:
        bounds = max_bounds_val
    else:
        bounds = min_bounds_val

    if max_res:
        res = max_res_val
    elif min_res:
        res = min_res_val
    else:
        res = max_res_val

    datasets = []

    for index, file_name in enumerate(dem_files):
        src = gu.georaster.Raster(file_name)

        if replace_fill_value_with_nan:
            src.data = replace_and_fill_nodata_value(src.data, src.nodata, np.nan)
            src.nodata = np.nan

        src = src.reproject(
            dst_crs=src.crs,
            nodata=src.nodata,
            dst_bounds=bounds,
            dst_res=res,
            resampling=resampling,
        )

        da = src.to_xarray(name="elevation")
        da = da.sel(band=1)
        del da.coords["band"]
        ds = da.to_dataset()

        if compute_hillshade and replace_fill_value_with_nan:
            hs_da = da.copy()
            hs_da.name = "hillshade"
            hs_da.values = calculate_hillshade(hs_da.values)

            ds["hillshade"] = hs_da

        datasets.append(ds)

        if save_to_nc:
            if nc_out_dir:
                out_fn = os.path.join(
                    nc_out_dir, str(pathlib.Path(file_name).stem) + ".nc"
                )

            else:
                out_fn = str(pathlib.Path(file_name).with_suffix("")) + ".nc"
            ds = ds.assign_coords({"time": times[index]})
            ds = ds.expand_dims("time")
            ds.to_netcdf(out_fn)

    ds = xr.concat(datasets, dim="time", combine_attrs="no_conflicts")
    ds = ds.assign_coords({"time": times})

    return ds


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
        elevations, no_data_value = geoprofile.core.sample_dem(
            dem_file, dense_coordinates
        )
        df[base_name] = elevations
        df.loc[
            df[base_name] == no_data_value, base_name
        ] = np.nan  # replace no data values with nan

    return df


def get_min_max_bounds_res(dem_files):
    """
    Gets min and max bounding box and resolution from list of input DEM files.

    Parameters
    ----------
    dem_files : [str,]
        List of string file paths.

    Returns
    -------
    min_bounds : rasterio.coords.BoundingBox
    max_bounds : rasterio.coords.BoundingBox
    min_res : float
    max_res : float
    """

    bounds_dict = {}
    res_dict = {}
    for f in dem_files:
        src = gu.georaster.Raster(f)
        bounds_dict[f] = src.bounds
        res_dict[f] = src.res

    max_bounds = bounds_dict[max(bounds_dict, key=bounds_dict.get)]
    min_bounds = bounds_dict[min(bounds_dict, key=bounds_dict.get)]

    max_res = res_dict[max(res_dict, key=res_dict.get)]
    min_res = res_dict[min(res_dict, key=res_dict.get)]

    return min_bounds, max_bounds, min_res, max_res


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


def gv_plot_dem_ds(
    ds,
    elevation_name="elevation",
    elevation_cmap="viridis",
    hillshade_name=None,
    hillshade_cmap="greys",
    hillshade_alpha=0.5,
    basemap=True,
    plot_size=200,
):
    gv.output(size=plot_size)
    gv.Dimension.type_formatters[np.datetime64] = "%Y-%m-%d"

    crs = cartopy.crs.epsg(ds.rio.crs.to_epsg())

    gv_elements = []

    elevation_img = (
        gv.Dataset(ds[elevation_name], crs=crs)
        .to(gv.Image, ["x", "y"], dynamic=True)
        .opts(tools=["hover"])
    )
    elevation = regrid(elevation_img.opts(colorbar=True, cmap=elevation_cmap))

    gv_elements.append(elevation)

    if isinstance(hillshade_name, type("")):
        hillshade_img = (
            gv.Dataset(ds[hillshade_name], crs=crs)
            .to(gv.Image, ["x", "y"], dynamic=True)
            .opts(alpha=hillshade_alpha)
        )
        hillshade = regrid(hillshade_img.opts(colorbar=True, cmap=hillshade_cmap))
        gv_elements.append(hillshade)

    if basemap:
        url = "https://mt1.google.com/vt/lyrs=s&x={X}&y={Y}&z={Z}"
        tiles = gv.WMTS(url, crs=crs)
        gv_elements.append(tiles)

    return functools.reduce(operator.mul, gv_elements)


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
    if np.isnan(nodata_value):
        masked_array = np.nan_to_num(array, nan=fill_value)
    else:
        mask = array == nodata_value
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


def xr_read_tif(tif_file_path, chunks=1000, masked=True):
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

    da = rioxarray.open_rasterio(tif_file_path, chunks=chunks, masked=True)

    # Extract bands and assign as variables in xr.Dataset()
    ds = xr.Dataset()
    for i, v in enumerate(da.band):
        da_tmp = da.sel(band=v)
        da_tmp.name = "band" + str(i + 1)

        ds[da_tmp.name] = da_tmp

    # Delete empty band coordinates.
    # Need to preserve spatial_ref coordinate, even though it appears empty.
    # See spatial_ref attributes under ds.coords.variables used by rioxarray extension.
    del ds.coords["band"]

    # Preserve top-level attributes and extract single value from value iterables e.g. (1,) --> 1
    ds.attrs = da.attrs
    for key, value in ds.attrs.items():
        try:
            if len(value) == 1:
                ds.attrs[key] = value[0]
        except TypeError:
            pass

    return ds
