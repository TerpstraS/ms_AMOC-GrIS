#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 04/01/2025
# ---------------------------------------------------------------------------
"""utils.py
Some utility functions

Necessary packages:
- numpy
- xarray
- pyproj
"""
# ---------------------------------------------------------------------------
import os
import numpy as np
import xarray as xr
import pyproj


def create_grl_atm_mask(lons, lats):
    """
    Create a mask for the atmospheric grid cells that overlap with Greenland.
    The mask is based on the geometry of Greenland from Natural Earth and the grid defined by the
    given longitude and latitude coordinates.
    A cell is masked if at least 30% of its area overlaps with Greenland.

    """
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import box

    # Build Greenland geometry from Natural Earth
    shpfilename = shpreader.natural_earth(
        resolution="10m", category="cultural", name="admin_0_countries"
    )
    reader = shpreader.Reader(shpfilename)
    greenland_geom = next(
        record.geometry for record in reader.records() if record.attributes["NAME"] == "Greenland"
    )

    lons = lons.values
    lats = lats.values


    def center_to_edges(values):
        """Convert center coordinates to edge coordinates for calculating overlap."""
        values = np.asarray(values, dtype=float)
        if values.size == 1:
            return np.array([values[0] - 0.5, values[0] + 0.5])

        mids = 0.5 * (values[:-1] + values[1:])
        first = values[0] - (mids[0] - values[0])
        last = values[-1] + (values[-1] - mids[-1])
        return np.concatenate(([first], mids, [last]))


    lon_edges = center_to_edges(lons)
    lat_edges = center_to_edges(lats)

    # Masking cells with minimum fraction overlapping Greenland
    min_overlap_fraction = 0.3
    mask = np.zeros((lats.size, lons.size), dtype=bool)
    for j in range(lats.size):
        for i in range(lons.size):
            cell_poly = box(
                min(lon_edges[i], lon_edges[i + 1]),
                min(lat_edges[j], lat_edges[j + 1]),
                max(lon_edges[i], lon_edges[i + 1]),
                max(lat_edges[j], lat_edges[j + 1]),
            )
            overlap_area = greenland_geom.intersection(cell_poly).area
            overlap_fraction = overlap_area / cell_poly.area if cell_poly.area > 0 else 0.0
            mask[j, i] = (overlap_fraction >= min_overlap_fraction)

    # Remove northern row, since it goes to far north
    mask[-1, :] = False

    mask_da = xr.DataArray(mask, dims=["lat", "lon"], coords={"lat": lats, "lon": lons})

    return mask_da


def round_away_from_zero(x):
    """Round a number away from zero.
    This function rounds positive numbers up and negative numbers down.

    Parameters
    ----------
    x : float
        The number to be rounded.

    Returns
    -------
    int
        The rounded integer value of the input number.
    """
    if x > 0:
        return int(x) + (x != int(x))
    elif x < 0:
        return int(x) - (x != int(x))
    else:
        return 0
    

def smooth_convolve(x, n=5, mode="valid", pad_mode=None):
    """Smooth the input array using a moving average filter.

    Parameters
    ----------
    x : np.ndarray
        Input array to be smoothed.
    n : int, optional
        Window size for the moving average filter, by default 5.
    mode : str, optional
        Convolution mode, by default "valid".
    pad_mode : str, optional
        Padding mode for the input array, by default None. If None, no padding is applied.

    Returns
    -------
    np.ndarray
        Smoothed array.
    """

    # padding
    if pad_mode is not None:
        pad_left = (n - 1) // 2
        pad_right = n // 2
        x = np.pad(x, (pad_left, pad_right), mode=pad_mode)

    return np.convolve(x, np.ones(n)/n, mode=mode)


def get_month(month_ind):
    """Get the month name from the month index.

    Parameters
    ----------
    month_ind : int
        Month index (1-13). If 13, it represents yearly data.
        
    Returns
    -------
    str
        Month name.
    """
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
        "Yearly"
    ]
    return months[month_ind - 1]


def transform_GIS_to_degrees(ds):
    """
    Transform the dataset coordinates from km to degrees.
    This includes changing xc and yc to degrees.

    This is valid for the GIS specification of the CLIMBER-X model with yelmo.
    """

    # Projection details from metadata
    stere_proj = pyproj.Proj(proj='stere',
                        lat_0=90.0,    # grid_phi
                        lon_0=-45.0,   # grid_lambda
                        lat_ts=70.0,   # true scale latitude
                        x_0=0, y_0=0,
                        ellps='WGS84')

    # Create transformer from stereographic to geographic
    transformer = pyproj.Transformer.from_proj(stere_proj, "epsg:4326", always_xy=True)

    # Convert km to m
    x = ds['xc'].values * 1000
    y = ds['yc'].values * 1000

    # Meshgrid if 1D coords
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)

    # Transform to lon/lat
    lon, lat = transformer.transform(x, y)

    # Add to dataset as coordinates
    ds = ds.assign_coords(
        lon=(("yc", "xc"), lon),
        lat=(("yc", "xc"), lat)
    )

    return ds


def transform_GIS_SMB_to_degrees(ds):
    """
    Transform the dataset coordinates from km to degrees.
    This includes changing xc and yc to degrees.

    This is valid for the GIS specification of the CLIMBER-X model with yelmo.
    """

    stere_proj = pyproj.Proj(proj='stere',
                        lat_0=90.0,    # grid_phi
                        lon_0=-45.0,   # grid_lambda
                        lat_ts=70.0,   # true scale latitude
                        x_0=0, y_0=0,
                        ellps='WGS84')

    # Create transformer from stereographic to geographic
    transformer = pyproj.Transformer.from_proj(stere_proj, "epsg:4326", always_xy=True)

    # Convert km to m
    x = ds['x'].values * 1000
    y = ds['y'].values * 1000

    # Shift x and y to the center of the grid 
    # (values taken from the CLIMBER-X GIS grid definition), since those start at the 
    # actual kilometer coordinates of the grid instead of at 0.
    x = x - 720000.06
    y = y - 3450000.2

    # Meshgrid if 1D coords
    if x.ndim == 1 and y.ndim == 1:
        x, y = np.meshgrid(x, y)

    # Transform to lon/lat
    lon, lat = transformer.transform(x, y)

    # Add to dataset as coordinates
    ds = ds.assign_coords(
        lon=(("y", "x"), lon),
        lat=(("y", "x"), lat)
    )

    return ds


def xr_concat_time_continuous(ds_list):
    """Concatenate a list of xarray datasets along the time dimension.
    This assumes each dataset to follow each other in time and that the time in each dataset
    starts at 0. This function makes the time increase continuously over the datasets.

    Parameters
    ----------
    ds_list : list
        List of xarray datasets.

    Returns
    -------
    xr.Dataset
        Concatenated xarray dataset.
    """
    if len(ds_list) == 1:
        return ds_list[0]
    else:
        for i in range(1, len(ds_list)):

            ds_list[i] = ds_list[i].assign_coords(
                time=ds_list[i].time + ds_list[i - 1].time[-1] + 1
            )

        # Concatenate along the time dimension
        combined_ds = xr.concat(ds_list, dim="time")

    return combined_ds


def xr_load_and_concat_time_continuous(experiments, DIR_DATA=None, dataset=None):
    """Concatenate a list of xarray datasets along the time dimension.
    This assumes each dataset to follow each other in time and that the time in each dataset
    starts at 0. This function makes the time increase continuously over the datasets.
    The function first loads the datasets with the full paths defined in the experiments list.

    Parameters
    ----------

    experiments : list
        List of file paths to xarray datasets. Can be relative to the directory DIR_DATA.
    DIR_DATA : str (optional)
        Directory where the data is stored. If None, experiments should contain full paths.
    dataset : str (optional)
        Name of the dataset to load from each experiment. If None, dataset should be in the file path
        of each experiment already

    Returns
    -------
    xr.Dataset
        Concatenated xarray dataset.
    """
    if len(experiments) == 0:
        raise ValueError("No experiments given.")

    if DIR_DATA is not None:
        experiments = [os.path.join(DIR_DATA, exp) for exp in experiments]

    ds_list = []
    for i, exp in enumerate(experiments):

        if dataset is not None:
            ds_list.append(xr.open_dataset(os.path.join(exp, dataset), mode="r"))
        else:
            ds_list.append(xr.open_dataset(exp, mode="r"))

    # Concatenate along the time dimension
    combined_ds = xr_concat_time_continuous(ds_list)

    return combined_ds
