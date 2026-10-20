#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 19/12/2023
# ---------------------------------------------------------------------------
"""plotting.py
Some common personal plot functionality

Necessary packages:
- plt
- numpy
- nc_time_axis
- cartopy
"""
# ---------------------------------------------------------------------------
import os
from functools import lru_cache
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import cartopy.mpl.ticker as cticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

import cmocean

import pyproj
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom

import nc_time_axis

from src.config import *
import src.utils as utils

# plt.rc("font", family="serif", size="medium")
# plt.rc("savefig", dpi=300, format="pdf")
# plt.rc("legend", loc="best", fontsize="small", fancybox=True, framealpha=1)
# plt.rc("lines", linewidth=1.8, markersize=6, markeredgewidth=1)
# plt.rc("axes.spines", top=False, right=False)
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Set plot style
style_path = os.path.join(DIR_SRC, "paper.mplstyle")
plt.style.use(style_path)

# Update parameters that cannot be set in the style file
# Reduce cbar fontsize
# plt.rcParams['axes.labelsize'] = 6
# plt.rcParams['xtick.labelsize'] = 6
# plt.rcParams['ytick.labelsize'] = 6


jour_sizes = {
    "default": {"onecol": 3.42, "twocol": 7.2},
    # Add more journals below. Can add more properties to each journal
}
golden = (1 + 5 ** 0.5) / 2

width_onecol = jour_sizes["default"]["onecol"]
height_onecol = width_onecol/golden

width_twocol = jour_sizes["default"]["twocol"]
height_twocol = width_twocol/golden

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Default projections and extents used across functions
DEFAULT_GIS_PROJECTION = ccrs.Stereographic(central_longitude=-42.75, central_latitude=71.5)
DEFAULT_YELMO_PROJECTION = ccrs.Stereographic(central_longitude=-45, central_latitude=90, true_scale_latitude=70)
DEFAULT_WORLD_PROJECTION = ccrs.Robinson()
DEFAULT_GIS_EXTENT = [-56, -29.5, 58, 85]


def get_color(i):
    return COLORS[i % len(COLORS)]


def _flatten_axes(axs):
    if isinstance(axs, np.ndarray):
        return axs.flatten()
    return [axs]


def _copy_cmap(cmap):
    if hasattr(cmap, "copy"):
        return cmap.copy()
    return cmap


def _get_data_array(ds, var=None):
    """Return a DataArray from either a DataArray or Dataset."""
    if isinstance(ds, xr.DataArray):
        return ds
    elif isinstance(ds, xr.Dataset):
        if var is None:
            raise ValueError("var must be specified when ds is an xarray Dataset")
        return ds[var]
    raise ValueError("ds must be an xarray DataArray or Dataset")


def _to_scalar(value):
    """Convert an xarray scalar (DataArray with 0 dims) to a Python float."""
    if isinstance(value, (xr.DataArray, xr.Variable)):
        return float(value.values)
    return float(value)


@lru_cache(maxsize=1)
def _get_yelmo_transformer():
    crs_xy = pyproj.CRS.from_proj4(
        "+proj=stere "
        "+lat_0=90 "
        "+lat_ts=70 "
        "+lon_0=-45 "
        "+k_0=1 "
        "+units=m "
        "+datum=WGS84"
    )
    crs_ll = pyproj.CRS.from_epsg(4326)
    return pyproj.Transformer.from_crs(crs_xy, crs_ll, always_xy=True)


@lru_cache(maxsize=1)
def _get_greenland_canada_geometries():
    shp = shpreader.natural_earth(
        resolution='50m',
        category='cultural',
        name='admin_0_countries'
    )

    greenland_geom = None
    canada_geom = None

    for record in shpreader.Reader(shp).records():
        if record.attributes['NAME'] == 'Greenland':
            greenland_geom = record.geometry
            if canada_geom is not None:
                break
        elif record.attributes['NAME'] == 'Canada':
            canada_geom = record.geometry
            if greenland_geom is not None:
                break

    return greenland_geom, canada_geom


def _resolve_mask(mask, ds):
    if mask is None:
        return None

    if isinstance(mask, str):
        if not isinstance(ds, xr.Dataset):
            raise ValueError("String mask requires ds to be an xarray Dataset")
        if mask == "mask":
            mask = ds[mask]
        else:
            mask = ds.isel(time=0)[mask] > 0
    elif isinstance(mask, xr.Dataset):
        mask = mask["mask"]
    elif isinstance(mask, (xr.DataArray, np.ndarray)):
        pass
    else:
        raise ValueError("mask must be str, xarray DataArray/Dataset, or numpy array")

    if isinstance(mask, xr.DataArray) and "time" in mask.dims:
        mask = mask.isel(time=0)

    return mask


def get_unit_circle():
    """
    Compute a circle in axes coordinates, which we can use as a boundary
    for the map. We can pan/zoom as much as we like - the boundary will be
    permanently circular.

    Code obtained from:
    https://scitools.org.uk/cartopy/docs/latest/gallery/always_circular_stereo.html#sphx-glr-gallery-always-circular-stereo-py

    Source:
    https://github.com/UU-IMAU/Python-for-lunch-Notebooks/blob/master/PFL_08_plotting_advanced/Examples%20Cartopy%20Leo.ipynb
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    return circle


def set_circular_boundary(ax):
    """
    Source:https://github.com/UU-IMAU/Python-for-lunch-Notebooks/blob/master/PFL_08_plotting_advanced/Examples%20Cartopy%20Leo.ipynb
    """
    if ax is not None:
        ax.set_boundary(
            get_unit_circle(), transform=ax.transAxes
        )  # See notebook on coordinate transformations, PFL7


### Convenience function to create figures with plt.subplots()
def subplots(
    nrows=1, ncols=1, figsize_defaults=None, figsize=(width_onecol, height_onecol), projection=None, constrained_layout=True, **kwargs
):
    """
    Create a figure with subplots

    Parameters
    ----------
    nrows : int
        Number of rows
    ncols : int
        Number of columns
    figsize_defaults : str, optional
        If given, set figsize to onecol or twocol. If None, use the given figsize.
    figsize : tuple, optional
        Size of the figure, by default (10, 10)
    projection : cartopy.crs, optional
        Projection of the map, by default None

    Returns
    -------
    fig, axs
        Figure and axes
    """

    # If figsize_defaults is given, set figsize accordingly. Otherwise, use the given figsize.
    if figsize_defaults is not None:
        if figsize_defaults == "onecol":
            figsize = (width_onecol, height_onecol)
        elif figsize_defaults == "twocol":
            figsize = (width_twocol, height_twocol)
        else:
            raise ValueError("figsize_defaults must be 'onecol' or 'twocol'")

    subplot_kw = kwargs.pop("subplot_kw", None)
    if subplot_kw is None:
        subplot_kw = {}
    if projection is not None:
        subplot_kw = {**subplot_kw, "projection": projection}

    subplot_kwargs = {
        "figsize": figsize,
        "constrained_layout": constrained_layout,
        **kwargs,
    }
    if subplot_kw:
        subplot_kwargs["subplot_kw"] = subplot_kw

    fig, axs = plt.subplots(nrows, ncols, **subplot_kwargs)
    return fig, axs


def subplots_single_column(
    nrows=1, ncols=1, figsize=(width_onecol, height_onecol), projection=None, constrained_layout=True, **kwargs
):
    """
    DEPRECATED: Use subplots() instead
    Create a figure with subplots

    Parameters
    ----------
    figsize : tuple, optional
        Size of the figure, by default (10, 10)
    projection : cartopy.crs, optional
        Projection of the map, by default None

    Returns
    -------
    fig, axs
        Figure and axes
    """
    fig, axs = subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        projection=projection,
        constrained_layout=constrained_layout,
        **kwargs,
    )
    return fig, axs


def subplots_double_column(nrows=1, ncols=1, figsize=(width_twocol, height_twocol), projection=None, constrained_layout=True, **kwargs):
    """
    DEPRECATED: Use subplots() instead
    Create a figure with subplots

    Parameters
    ----------
    figsize : tuple, optional
        Size of the figure, by default (10, 10)
    projection : cartopy.crs, optional
        Projection of the map, by default None

    Returns
    -------
    fig, axs
        Figure and axes
    """
    fig, axs = subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        projection=projection,
        constrained_layout=constrained_layout,
        **kwargs,
    )
    return fig, axs


### Standard subplots for differerent regions
def subplots_GIS(
    nrows=1, ncols=1, figsize_defaults=None, figsize=(width_onecol, height_onecol), 
    projection=ccrs.Stereographic(central_longitude=-42.75, central_latitude=71.5),
    extent=[-56, -29.5, 58, 85],
    constrained_layout=True, **kwargs
):
    
    fig, axs = subplots(
        nrows=nrows,
        ncols=ncols,
        figsize_defaults=figsize_defaults,
        figsize=figsize,
        projection=projection,
        constrained_layout=constrained_layout,
        **kwargs,
    )

    # Apply default settings to all axes
    axs_flat = _flatten_axes(axs)
    for ax in axs_flat:

        ax.set_title("")

        ax.set_extent(extent, crs=ccrs.PlateCarree())
        # ax.coastlines(linewidth=1, edgecolor='black')

        # gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, xpadding=10)
        # gl.top_labels = False
        # gl.right_labels = False
        # gl.left_labels = False
        # gl.bottom_labels = False

    return fig, axs


def subplots_arctic(
    nrows=1, ncols=1, figsize_defaults=None, figsize=(width_onecol, height_onecol), 
    projection=ccrs.Orthographic(central_longitude=0, central_latitude=90),
    constrained_layout=True, **kwargs
):
    
    fig, axs = subplots(
        nrows=nrows,
        ncols=ncols,
        figsize_defaults=figsize_defaults,
        figsize=figsize,
        projection=projection,
        constrained_layout=constrained_layout,
        **kwargs,
    )

    # Apply default settings to all axes
    axs_flat = _flatten_axes(axs)
    for ax in axs_flat:

        ax.set_title("")

        # ax.coastlines(linewidth=1, edgecolor='black')

        set_circular_boundary(ax)

        # gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, xpadding=10)
        # gl.top_labels = False
        # gl.right_labels = False
        # gl.left_labels = False
        # gl.bottom_labels = False

    return fig, axs


def subplots_antarctic(
    nrows=1, ncols=1, figsize_defaults=None, figsize=(width_onecol, height_onecol), 
    projection=ccrs.Orthographic(central_longitude=0, central_latitude=-90),
    constrained_layout=True, **kwargs
):
    
    fig, axs = subplots(
        nrows=nrows,
        ncols=ncols,
        figsize_defaults=figsize_defaults,
        figsize=figsize,
        projection=projection,
        constrained_layout=constrained_layout,
        **kwargs,
    )

    # Apply default settings to all axes
    axs_flat = _flatten_axes(axs)
    for ax in axs_flat:

        ax.set_title("")

        # ax.coastlines(linewidth=1, edgecolor='black')

        set_circular_boundary(ax)

        # gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, xpadding=10)
        # gl.top_labels = False
        # gl.right_labels = False
        # gl.left_labels = False
        # gl.bottom_labels = False

    return fig, axs


def subplots_world(
    nrows=1, ncols=1, figsize_defaults=None, figsize=(width_onecol, height_onecol), 
    projection=ccrs.Robinson(),
    constrained_layout=True, **kwargs
):
    
    fig, axs = subplots(
        nrows=nrows,
        ncols=ncols,
        figsize_defaults=figsize_defaults,
        figsize=figsize,
        projection=projection,
        constrained_layout=constrained_layout,
        **kwargs,
    )

    # Apply default settings to all axes
    axs_flat = _flatten_axes(axs)
    for ax in axs_flat:

        ax.set_title("")

    return fig, axs


### Plotting functions for CLIMBER-X output

def ds_plot_line(
    ds,
    var,
    varx="time",
    scaling=None,
    offset=None,
    ds_diff=None,
    ax=None,
    xlabel="time (years)",
    ylabel=None,
    title=None,
    xlim="default",
    ylim="default",
    grid=True,
    savefig=False,
    format="png",
    **kwargs
):
    """
    Plot a line plot of a variable var with a given DataArray or DataSet ds for the GIS.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        DataArray or Dataset with the variable var
    var : str
        variable name
    varx : str, optional
        x-variable name, by default "time".
        Can only be different than time if ds is a DataSet containing the variable varx.
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        axes to plot on, by default None. If None, generates a new figure.
    xlabel : str, optional
        x-axis label, by default "time (years)"
    ylabel : str
        y-axis label, by default None. If None, uses var as ylabel
    title : str, optional
        title of the plot
    xlim : tuple, optional
        x-axis limits. If None, no limits are set. If "default", the min and max of the data are used.
        else, a tuple with the limits can be given.
    ylim : tuple, optional
        y-axis limits. If None, no limits are set. If "default", the min and max of the data are used.
        else, a tuple with the limits can be given.
    grid : bool, optional

    Returns
    -------
    fig, axs
        Figure and axes
    """

    # Check whether we have a DataArray or a DataSet
    if isinstance(ds, xr.DataArray):
        da = ds
    elif isinstance(ds, xr.Dataset):
        da = ds[var]
    else:
        raise ValueError("ds must be an xarray DataArray or DataSet")

    if isinstance(varx, str):
        try:
            data_x = ds[varx]
        except KeyError:
            raise KeyError(
                f"DataSet does not contain variable {varx} / ds has to be a DataArray for custom varx"
            )
    else: # Assume varx is np array or xarray DataArray
        data_x = varx

    if ds_diff is not None:
        if isinstance(ds_diff, xr.DataArray):
            da_diff = ds_diff
        elif isinstance(ds_diff, xr.Dataset):
            da_diff = ds_diff[var]
        else:
            raise ValueError("ds_diff must be an xarray DataArray or DataSet")

        # If length of data is not equal, remove excess data points from the end of the longer array
        # And subtract
        min_length = min(len(da), len(da_diff))
        da = da.isel({da.dims[0]: slice(0, min_length)})
        da_diff = da_diff.isel({da_diff.dims[0]: slice(0, min_length)})

        try:
            data_x = data_x.isel({data_x.dims[0]: slice(0, min_length)})
        except (AttributeError, IndexError):  # numpy array or plain sequence
            data_x = data_x[:min_length]

        da = da - da_diff

    if scaling is not None:
        da = da * scaling
    if offset is not None:
        da = da + offset

    if ax is None:
        fig, ax = subplots()
    else:
        fig = ax.figure

    ax.plot(data_x, da, **kwargs)
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ax.set_ylabel(var)
    else:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title, pad=20)

    if xlim is not None:
        if xlim == "default":
            ax.set_xlim(data_x.min(), data_x.max())
        else:
            ax.set_xlim(xlim)

    if ylim is not None:
        if ylim == "default":
            ax.set_ylim(da.min(), da.max())
        else:
            ax.set_ylim(ylim)

    if grid:
        ax.grid(True, alpha=0.5)

    if savefig:
        os.makedirs("figures", exist_ok=True)
        fpath = os.path.join("figures", f"lineplot_{var}.{format}")
        plt.savefig(fpath, format=format)

    return fig, ax


def ds_plot_AMOC_streamfunction(
    ds_ocn, time_ind=0, ds_diff=None, ax=None, vmin=None, vmax=None, nlevels=23, 
    add_colorbar=True, cbar_kwargs=None, **kwargs
):
    """Plot AMOC streamfunction from ocean dataset at given time index."""

    # mask where data equals zero and plot the masked values in grey
    # Use data a_atl for this (nan values indicate bathymetry)
    ds_a_atl = ds_ocn['a_atl'].isel(time=time_ind)
    nan_mask = ds_a_atl.isnull()
    ds_mask = xr.where(nan_mask, 1, np.nan)

    ds_ocn_plot = ds_ocn['opsi_a'].isel(time=time_ind, month=12).sel(latv1=slice(-30, 85)).copy()

    # If ds_diff is given, plot the difference
    if ds_diff is not None:
        ds_ocn_plot_diff = ds_diff['opsi_a'].isel(time=time_ind, month=12).sel(latv1=slice(-30, 85))
        ds_ocn_plot = ds_ocn_plot - ds_ocn_plot_diff

    # Rescale depth to km
    ds_ocn_plot['levw'] = ds_ocn_plot['levw'] / 1000.0  # .data / 1000.0  # in m
    ds_mask['lev'] = ds_mask['lev'] / 1000.0  # .data / 1000.0  # in m
    
    # Minimum and maximum values for the colorbar.
    # Round vmin and vmax to nearest higher absolute integer for better readiblity.
    print(f"vmin: {utils.round_away_from_zero(ds_ocn_plot.min().values)}")
    print(f"vmax: {utils.round_away_from_zero(ds_ocn_plot.max().values)}")

    if vmin is None and vmax is None:
        vmin = utils.round_away_from_zero(ds_ocn_plot.min().values)
        vmax = utils.round_away_from_zero(ds_ocn_plot.max().values)
        vmin = -max(abs(vmin), abs(vmax))
        vmax = max(abs(vmin), abs(vmax))
    
    levels = np.linspace(vmin, vmax, nlevels)

    if ax is None:
        fig, ax = subplots()
    else:
        fig = ax.figure

    if add_colorbar:
        if cbar_kwargs is None:
            cbar_kwargs = {"label": "AMOC strength (Sv)", "extend": "both"}
    else:
        cbar_kwargs = None
    ds_ocn_plot.plot.contourf(
        x='latv1', y='levw', ax=ax, vmin=vmin, vmax=vmax, cmap='RdBu_r', levels=levels, 
        zorder=1, add_colorbar=add_colorbar, cbar_kwargs=cbar_kwargs, **kwargs
    )
    
    ds_mask.plot.pcolormesh(x='lat', y='lev', ax=ax, cmap='Greys', vmin=0, vmax=5, alpha=1, add_colorbar=False, zorder=2)

    ax.invert_yaxis()
    ax.set_xlim(-30, 85)
    ax.set_ylim(5, 0)
    ax.set_xlabel('latitude')
    ax.set_ylabel('depth (km)')
    ax.set_title("")

    ax.grid(True, linewidth=0.8,
        color='gray',
        alpha=0.2,
        linestyle='--')

    # Format tick labels as degrees
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d°'))

    return fig,ax


def ds_plot_ocn_NA(*args, **kwargs):
    kwargs["extent"] = "NA"
    return ds_plot_pcolormesh(*args, **kwargs)


def ds_plot_pcolormesh(
    ds, var, ds_diff=None, time_ind=0, month_ind=12, time_ind_ds_diff=None, scaling=None, offset=None, ax=None, cmap=None, vmin=None, vmax=None, cmap_centered=False, nlevels=11, extent=None, 
    projection=ccrs.Robinson(), grid=True, add_colorbar=True, cbar_kwargs=None, **kwargs
):
    """Plot North Atlantic mixed layer depth at given time index."""

    # Check whether we have a DataArray or a DataSet
    if isinstance(ds, xr.DataArray):
        da = ds
    elif isinstance(ds, xr.Dataset):
        da = ds[var]
    else:
        raise ValueError("ds must be an xarray DataArray or DataSet")

    if ax is None:
        fig, ax = subplots(projection=projection)
    else:
        fig = ax.figure

    # Storing lon_slice and lat_slice as well, to be able to estimate vmin and vmax later
    if extent == "NA":
        ax.set_extent([-80, 20, 30, 85], crs=ccrs.PlateCarree())
        lon_slice = slice(-80, 20)
        lat_slice = slice(30, 85)
    elif extent == "global":
        lon_slice = slice(-180, 180)
        lat_slice = slice(-90, 90)
    elif extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        lon_slice = slice(extent[0], extent[1])
        lat_slice = slice(extent[2], extent[3])
    else:
        lon_slice = slice(-180, 180)
        lat_slice = slice(-90, 90)
    
    # Check if month or mon is in dataset coordinates/dimensions
    try:
        if 'month' in da.coords:
            ds_plot = da.isel(time=time_ind, month=month_ind)
        elif 'mon' in da.coords:
            ds_plot = da.isel(time=time_ind, mon=month_ind)
        else:
            print("No month coordinate found. Using time index only.")
            ds_plot = da.isel(time=time_ind)
    except ValueError:
        print("No time/month coordinate found.")
        pass
        
    # If ds_diff is given, plot the difference
    if ds_diff is not None and time_ind_ds_diff is None:
        time_ind_ds_diff = time_ind

    if ds_diff is not None:
        if isinstance(ds_diff, xr.DataArray):
            da_diff = ds_diff
        elif isinstance(ds_diff, xr.Dataset):
            da_diff = ds_diff[var]
        else:
            raise ValueError("ds_diff must be an xarray DataArray or DataSet")

        if 'month' in da_diff.coords:
            ds_plot_diff = da_diff.isel(time=time_ind_ds_diff, month=month_ind)
        elif 'mon' in da_diff.coords:
            ds_plot_diff = da_diff.isel(time=time_ind_ds_diff, mon=month_ind)
        else:
            print("No month coordinate found in ds_diff. Using time index only.")
            ds_plot_diff = da_diff.isel(time=time_ind_ds_diff)
        ds_plot = ds_plot - ds_plot_diff
    
    if scaling is not None:
        ds_plot = ds_plot * scaling
    if offset is not None:
        ds_plot = ds_plot + offset
    
    vmin_print = ds_plot.sel(lon=lon_slice, lat=lat_slice).min().values
    vmax_print = ds_plot.sel(lon=lon_slice, lat=lat_slice).max().values
    print(f"vmin: {vmin_print}, {utils.round_away_from_zero(vmin_print)}")
    print(f"vmax: {vmax_print}, {utils.round_away_from_zero(vmax_print)}")

    if vmin is None:
        vmin = utils.round_away_from_zero(ds_plot.min().values)
    if vmax is None:
        vmax = utils.round_away_from_zero(ds_plot.max().values)

    if cmap_centered:
        vmin = -max(abs(vmin), abs(vmax))
        vmax = max(abs(vmin), abs(vmax))
    
    levels = np.linspace(vmin, vmax, nlevels)

    # Select colormap. Set land cells (nan) to "tan" color
    if cmap is None:
        cmap = cmocean.cm.deep
    cmap = _copy_cmap(cmap)
    cmap.set_bad(color='tan') 

    if add_colorbar:
        if cbar_kwargs is None:
            cbar_kwargs = {"label": var, "extend": "neither"}
    else:
        cbar_kwargs = None

    ds_plot.plot.pcolormesh(
        x='lon', y='lat', ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, levels=levels, transform=ccrs.PlateCarree(),
        zorder=1, add_colorbar=add_colorbar, cbar_kwargs=cbar_kwargs, **kwargs
    )

    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    ax.set_title("")
    
    if grid:
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.8,
            color='gray',
            alpha=0.2,
            linestyle='--'
        )

        # Manually set tick locations
        # gl.xlocator = mticker.FixedLocator([-100, -50, 0])
        # gl.ylocator = mticker.FixedLocator([-80, -40, 0, 40, 80])

        # Format tick labels as degrees
        gl.xformatter = cticker.LongitudeFormatter()
        gl.yformatter = cticker.LatitudeFormatter()

        # Label styling
        gl.xlabel_style = {"size": 6}
        gl.ylabel_style = {"size": 6}

        # Only label left & bottom
        gl.top_labels = False
        gl.right_labels = False
        
    return fig, ax


def ds_plot_contourf():
    pass


# Plotting functions
def plot_line(
    x,
    y,
    ax=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim="default",
    ylim="default",
    grid=False,
    **kwargs,
):
    """
    Plot a line plot.

    Parameters
    ----------
    x : array-like
    y : array-like
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        axes to plot on, by default None. If None, generates a new figure.
    xlabel : str, optional
        x-axis label, by default "time (years)"
    ylabel : str
        y-axis label, by default None. If None, uses var as ylabel
    title : str, optional
        title of the plot
    xlim : tuple, optional
        x-axis limits. If None, no limits are set. If "default", the min and max of the data are used.
        else, a tuple with the limits can be given.
    ylim : tuple, optional
        y-axis limits. If None, no limits are set. If "default", the min and max of the data are used.
        else, a tuple with the limits can be given.
    grid : bool, optional
    **kwargs : dict
        Additional keyword arguments for the plt.plot() function

    Returns
    -------
    fig, axs
        Figure and axes
    """

    if ax is None:
        fig, ax = subplots_single_column()
    else:
        fig = ax.figure

    ax.plot(x, y, **kwargs)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title, pad=20)

    if xlim is not None:
        if xlim == "default":
            ax.set_xlim(min(x), max(x))
        else:
            ax.set_xlim(xlim)

    if ylim is not None:
        if ylim == "default":
            ax.set_ylim(min(y), max(y))
        else:
            ax.set_ylim(ylim)

    if grid:
        ax.grid(True, alpha=0.5)

    return fig, ax


def plot_GIS_contourf_yelmo(
    ds,
    var,
    time_index=None,
    mask=None,
    ax=None,
    vmin=None,
    vmax=None,
    levels=None,
    cbar_label=None,
    cmap=None,
    title="time",
    savefig=False,
    format="png",
    fname_processed_data=None,
    load_processed_data=False,
):
    """
    Plot a contour plot of a variable var with a given DataArray or DataSet ds at a given time index.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        DataArray or Dataset with the variable var
    var : str
        variable name
    time_index : int
        time index of variable to plot, by default None.
        If no time axis exists, select None. If None and time axis exists, defaults to time index 1
    mask : xarray.DataArray, optional
        mask to apply to the data, by default None. Should have the same spatial dimensions as var.
        If ds is a DataSet, mask can also be the string "mask" to use the mask in the DataSet
        (if it exists). If mask is any other string, assume it is a variable in the DataSet and use
        that as the mask (by masking for values > 0 and taking first time index).
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        axes to plot on, by default None. If None, generates a new figure. Existing ax should have
        projection: ccrs.Stereographic(central_longitude=-45, central_latitude=70)

    vmin : float, optional
        vmin
    vmax : float, optional
        vmax
    levels : int, optional
        number of levels of contourf
    cbal_label : str, optional
        colorbar label, by default None (defaults to var)
    title : str, optional
        title of the plot, by default "time" (gives time in years as title)

    Returns
    -------
    fig, axs
        Figure and axes
    """

    # Check if preprocessed data file exists
    if load_processed_data and fname_processed_data is not None and os.path.exists(fname_processed_data):

        # Load as DataArray from the processed data file
        da = xr.open_dataarray(fname_processed_data)

    else:

        # Check whether we have a DataArray or a DataSet for the variable var
        if isinstance(ds, xr.DataArray):
            da = ds
        elif isinstance(ds, xr.Dataset):
            da = ds[var]
        else:
            raise ValueError("ds must be an xarray DataArray or DataSet")

        # Select time index
        if time_index is not None and "time" in da.dims:
            da = da.isel(time=time_index)
        else:
            # Check if there is time dimension, if so, take the first time index
            if "time" in da.dims:
                da = da.isel(time=0)

        # print(da)
        # print(mask)
    

        # Check whether we have a DataArray or a DataSet for the mask
        if mask is not None:
            mask = _resolve_mask(mask, da)
            da = da.where(mask)

        # Remove values directly around zero, because it is negligible/model artifact of slightly 
        # negative thickness. Applying the mask with xarray preserves the DataArray dimensions and 
        # coordinates for plotting.
        da = da.where((da <= -1.0) | (da >= 1.0))
        da = da / 1000.0  # Convert from m to km

        # Save da to data/plotting in case full data is not available (save as dataarray)
        if fname_processed_data is not None:
            da.to_netcdf(fname_processed_data)

    # Create mask for values outside greenland mainland northwest side, using xc and yc coordinates
    # mask_northwest = (da["xc"] < -350) & (da["yc"] > -1100)
    # da = da.where(~mask_northwest)

    if ax is None:
        # Set up the plot with a projection (e.g., PlateCarree for regular lat/lon data)
        fig, ax = plt.subplots(
            subplot_kw={
                "projection": ccrs.Stereographic(central_longitude=-45, central_latitude=90, true_scale_latitude=70)
            }
        )
    else:
        fig = ax.figure

    # Transform coordinates from model grid to lon/lat
    xc = ds["xc"].values * 1000 # meters
    yc = ds["yc"].values * 1000 # meters

    xc_2d, yc_2d = np.meshgrid(xc, yc)

    transformer = _get_yelmo_transformer()
    lon, lat = transformer.transform(xc_2d, yc_2d)
    
    ax.set_extent([-57, -30, 59, 84], crs=ccrs.PlateCarree())

    # Create cmap with white values under vmin
    if cmap is None:
        cmap = _copy_cmap(cmocean.cm.ice)
        cmap.set_under(color='white')  # Set color for values below vmin

    # Label of colorbar
    if cbar_label is None:
        cbar_label = var

    if levels is None:
        levels = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    elif levels == "default":
        levels = None

    cf = ax.contourf(
        lon,
        lat,
        da,
        transform=ccrs.PlateCarree(),  # Transform from data's coordinate system
        levels=levels,
        cmap=cmap,
    )

    ax.axis("off")  # Turn off the axis

    if title == "time":
        if time_index is not None:
            ax.set_title(f"time = {time_index} years", fontweight='bold')
    elif title is not None:
        ax.set_title(title, fontsize=6, fontweight='bold')
    else:
        ax.set_title("", fontsize=6, fontweight='bold')
    
    greenland_geom, canada_geom = _get_greenland_canada_geometries()

    # Now draw only Greenland's outline
    # ax.add_geometries(
    #     [greenland_geom, canada_geom],
    #     crs=ccrs.PlateCarree(),
    #     facecolor='none',
    #     edgecolor='black',
    #     linewidth=0.5,
    #     zorder=10
    # )
    ax.add_geometries(
        [greenland_geom],
        crs=ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black',
        linewidth=0.5,
        zorder=10
    )


    if savefig:
        # os.makedirs("figures", exist_ok=True)
        fpath = os.path.join("figures", f"GIS_contourf_{var}_timeindex-{time_index}.{format}")
        plt.savefig(fpath, format=format)

    return fig, ax, cf


def plot_GIS_smb_contourf_yelmo(
    ds,
    var,
    time_index=None,
    mask=None,
    ax=None,
    vmin=None,
    vmax=None,
    levels=None,
    cbar_label=None,
    cmap=None,
    title="time",
    savefig=False,
    format="png",
    fname_processed_data=None,
    load_processed_data=False,
):
    """
    Plot a contour plot of a variable var with a given DataArray or DataSet ds at a given time index.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        DataArray or Dataset with the variable var
    var : str
        variable name
    time_index : int
        time index of variable to plot, by default None.
        If no time axis exists, select None. If None and time axis exists, defaults to time index 1
    mask : xarray.DataArray, optional
        mask to apply to the data, by default None. Should have the same spatial dimensions as var.
        If ds is a DataSet, mask can also be the string "mask" to use the mask in the DataSet
        (if it exists). If mask is any other string, assume it is a variable in the DataSet and use
        that as the mask (by masking for values > 0 and taking first time index).
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        axes to plot on, by default None. If None, generates a new figure. Existing ax should have
        projection: ccrs.Stereographic(central_longitude=-45, central_latitude=70)

    vmin : float, optional
        vmin
    vmax : float, optional
        vmax
    levels : int, optional
        number of levels of contourf
    cbal_label : str, optional
        colorbar label, by default None (defaults to var)
    title : str, optional
        title of the plot, by default "time" (gives time in years as title)

    Returns
    -------
    fig, axs
        Figure and axes
    """

    # Check if preprocessed data file exists
    if load_processed_data and fname_processed_data is not None and os.path.exists(fname_processed_data):

        # Load as DataArray from the processed data file
        da = xr.open_dataarray(fname_processed_data)

    else:

        # Check whether we have a DataArray or a DataSet for the variable var
        if isinstance(ds, xr.DataArray):
            da = ds
        elif isinstance(ds, xr.Dataset):
            da = ds[var]
        else:
            raise ValueError("ds must be an xarray DataArray or DataSet")

        # Select time index
        if time_index is not None and "time" in da.dims:
            da = da.isel(time=time_index)
        else:
            # Check if there is time dimension, if so, take the first time index
            if "time" in da.dims:
                da = da.isel(time=0)

        # Check whether we have a DataArray or a DataSet for the mask
        if mask is not None:
            mask = _resolve_mask(mask, da)
            da = da.where(mask)

        # Save da to data/plotting in case full data is not available (save as dataarray)
        if fname_processed_data is not None:
            da.to_netcdf(fname_processed_data)

        
    # Create mask for values outside greenland mainland northwest side, using xc and yc coordinates
    # mask_northwest = (da["xc"] < -350) & (da["yc"] > -1100)
    # da = da.where(~mask_northwest)

    if ax is None:
        # Set up the plot with a projection (e.g., PlateCarree for regular lat/lon data)
        fig, ax = plt.subplots(
            subplot_kw={
                "projection": ccrs.Stereographic(central_longitude=-45, central_latitude=90, true_scale_latitude=70)
            }
        )
    else:
        fig = ax.figure

    # Transform coordinates from model grid to lon/lat
    xc = (da["x"].values - 720) * 1000 # meters
    yc = (da["y"].values - 3450) * 1000 # meters

    da = (
        da.assign_coords(
            x=("x", da["x"].values),
            y=("y", da["y"].values),
        )
        .rename({"x": "xc", "y": "yc"})
    )

    xc_2d, yc_2d = np.meshgrid(xc, yc)

    transformer = _get_yelmo_transformer()
    lon, lat = transformer.transform(xc_2d, yc_2d)
    
    ax.set_extent([-57, -30, 59, 84], crs=ccrs.PlateCarree())

    # Label of colorbar
    if cbar_label is None:
        cbar_label = var

    if cmap is None:
        cmap = _copy_cmap(plt.get_cmap('RdBu_r'))
        cmap.set_under(color='white')  # Set color for values below vmin

    # levels = np.array([-2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000])
    if levels is None:
        levels = np.arange(-2000, 2500, 125)
    elif levels == "default":
        levels = None

    cf = ax.contourf(
        lon,
        lat,
        da,
        transform=ccrs.PlateCarree(),  # Transform from data's coordinate system
        levels=levels,
        cmap=cmap,
    )

    ax.axis("off")  # Turn off the axis

    if title == "time":
        if time_index is not None:
            ax.set_title(f"time = {time_index} years", fontweight='bold')
    elif title is not None:
        ax.set_title(title, fontsize=6, fontweight='bold')
    else:
        ax.set_title("", fontsize=6, fontweight='bold')
    
    greenland_geom, canada_geom = _get_greenland_canada_geometries()

    # Now draw only Greenland's outline
    # ax.add_geometries(
    #     [greenland_geom, canada_geom],
    #     crs=ccrs.PlateCarree(),
    #     facecolor='none',
    #     edgecolor='black',
    #     linewidth=0.5,
    #     zorder=10
    # )
    ax.add_geometries(
        [greenland_geom],
        crs=ccrs.PlateCarree(),
        facecolor='none',
        edgecolor='black',
        linewidth=0.5,
        zorder=10
    )


    if savefig:
        # os.makedirs("figures", exist_ok=True)
        fpath = os.path.join("figures", f"GIS_contourf_{var}_timeindex-{time_index}.{format}")
        plt.savefig(fpath, format=format)

    return fig, ax, cf


def plot_GIS_contourf_atmos(
    ds,
    var,
    time_index=None,
    month_index=None,
    extent=None,
    ax=None,
    vmin=None,
    vmax=None,
    levels=10,
    cbar_kwargs=None,
    norm=False,
    title="time",
    projection=None,
    cmap=plt.cm.viridis,
    ds_control=None,
    control_method="anomaly",
    zorder=10,
):
    """
    DEPRECATED: Use subplots_GIS_atmos() instead
    Plot a contour plot of a variable var with a given DataArray or DataSet ds at a given time index.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        DataArray or Dataset with the variable var
    var : str
        variable name
    time_index : int
        time index of variable to plot, by default None.
        If no time axis exists, select None. If None and time axis exists, defaults to time index 1
    mask : xarray.DataArray, optional
        mask to apply to the data, by default None. Should have the same spatial dimensions as var.
        If ds is a DataSet, mask can also be the string "mask" to use the mask in the DataSet
        (if it exists). If mask is any other string, assume it is a variable in the DataSet and use
        that as the mask (by masking for values > 0 and taking first time index).
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        axes to plot on, by default None. If None, generates a new figure. Existing ax should have
        projection: ccrs.Stereographic(central_longitude=-45, central_latitude=70)

    vmin : float, optional
        vmin
    vmax : float, optional
        vmax
    levels : int, optional
        number of levels of contourf
    cbal_label : str, optional
        colorbar label, by default None (defaults to var)
    title : str, optional
        title of the plot, by default "time" (gives time in years as title)

    Returns
    -------
    fig, axs
        Figure and axes
    """

    # Check whether we have a DataArray or a DataSet for the variable var
    if isinstance(ds, xr.DataArray):
        da = ds
    elif isinstance(ds, xr.Dataset):
        da = ds[var]
    else:
        raise ValueError("ds must be an xarray DataArray or DataSet")

    # Subtract control simulation
    if ds_control is not None:
        if isinstance(ds_control, xr.DataArray):
            da_control = ds_control
        elif isinstance(ds_control, xr.Dataset):
            da_control = ds_control[var]
        else:
            raise ValueError("ds_control must be an xarray DataArray or DataSet")

        if control_method == "anomaly":
            da = da - da_control
        elif control_method == "percentage":
            da = (da - da_control) / da_control * 100

    # Select time index
    if time_index is not None:
        time_years = int(da.coords["time"][time_index])
        da = da.isel(time=time_index)
    else:
        # Check if there is time dimension, if so, take the first time index
        if "time" in da.dims:
            da = da.isel(time=0)
    if month_index is not None:
        try:
            da = da.isel(month=month_index)
        except ValueError:
            da = da.isel(mon=month_index)

    if ax is None:
        if projection is None:
            projection = ccrs.Robinson()
        # Set up the plot with a projection (e.g., PlateCarree for regular lat/lon data)
        fig, ax = subplots(projection=projection)
    else:
        fig = ax.figure

    # Create cmap with white values under vmin and above vmax
    cmap = _copy_cmap(cmap)
    if vmin is not None and not norm:
        cmap.set_under(color="white")
    if vmax is not None and not norm:
        cmap.set_over(color="white")

    # Label of colorbar
    if cbar_kwargs is None:
        cbar_kwargs = {"label": var, "shrink": 0.6}

    # min and max values of data in given extent
    if extent is not None:
        da_min = da.sel(lat=slice(extent[2], extent[3]), lon=slice(extent[0], extent[1])).min()
        da_max = da.sel(lat=slice(extent[2], extent[3]), lon=slice(extent[0], extent[1])).max()
    else:
        da_min = da.min()
        da_max = da.max()

    if norm:
        norm = mcolors.TwoSlopeNorm(vmin=min(-1e-9, da_min), vcenter=0, vmax=max(1e-9, da_max))
        vmin = None
        vmax = None
    else:
        norm = mcolors.Normalize(vmin=da_min, vmax=da_max)
        vmin = None
        vmax = None

    if extent is not None:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # plt.contourf(
    #     da.lon,
    #     da.lat,
    #     da,
    #     transform=ccrs.PlateCarree(),  # Transform from data's coordinate system
    #     cmap=cmap,  # Choose your color map
    #     vmin=vmin,
    #     vmax=vmax,
    #     levels=levels,
    # )

    cf = da.plot.contourf(
        ax=ax,
        transform=ccrs.PlateCarree(),  # Transform from data's coordinate system
        cmap=cmap,  # Choose your color map
        cbar_kwargs=cbar_kwargs,
        vmin=vmin,
        vmax=vmax,
        levels=levels,
        norm=norm
    )

    ax.coastlines()
    # ax.axis("off")
    # ax.grid(True)
    
    if title == "time":
        if time_index is not None:
            ax.set_title(f"time = {time_years} years", pad=20)
    elif title is not None:
        ax.set_title(title, pad=20)
    else:
        ax.set_title(None)

    return fig, ax, cf


def plot_GIS_contourf_ocn(
    ds,
    var,
    time_index=None,
    month_index=None,
    ax=None,
    vmin=None,
    vmax=None,
    levels=10,
    cbar_label=None,
    title="time",
    projection=None,
    cmap=plt.cm.viridis,
    ds_control=None,
):
    """
    Plot a contour plot of a variable var with a given DataArray or DataSet ds at a given time index.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
        DataArray or Dataset with the variable var
    var : str
        variable name
    time_index : int
        time index of variable to plot, by default None.
        If no time axis exists, select None. If None and time axis exists, defaults to time index 1
    mask : xarray.DataArray, optional
        mask to apply to the data, by default None. Should have the same spatial dimensions as var.
        If ds is a DataSet, mask can also be the string "mask" to use the mask in the DataSet
        (if it exists). If mask is any other string, assume it is a variable in the DataSet and use
        that as the mask (by masking for values > 0 and taking first time index).
    ax : matplotlib.axes._subplots.AxesSubplot, optional
        axes to plot on, by default None. If None, generates a new figure. Existing ax should have
        projection: ccrs.Stereographic(central_longitude=-45, central_latitude=70)

    vmin : float, optional
        vmin
    vmax : float, optional
        vmax
    levels : int, optional
        number of levels of contourf
    cbal_label : str, optional
        colorbar label, by default None (defaults to var)
    title : str, optional
        title of the plot, by default "time" (gives time in years as title)

    Returns
    -------
    fig, axs
        Figure and axes
    """

    # Check whether we have a DataArray or a DataSet for the variable var
    if isinstance(ds, xr.DataArray):
        da = ds
    elif isinstance(ds, xr.Dataset):
        da = ds[var]
    else:
        raise ValueError("ds must be an xarray DataArray or DataSet")

    # Subtract control simulation
    if ds_control is not None:
        if isinstance(ds_control, xr.DataArray):
            da_control = ds_control
        elif isinstance(ds_control, xr.Dataset):
            da_control = ds_control[var]
        else:
            raise ValueError("ds_control must be an xarray DataArray or DataSet")

        da = da - da_control

    # Select time index
    if time_index is not None:
        time_years = int(da.coords["time"][time_index])
        da = da.isel(time=time_index)
    else:
        # Check if there is time dimension, if so, take the first time index
        if "time" in da.dims:
            da = da.isel(time=0)
    if month_index is not None:
        da = da.isel(month=month_index)

    if ax is None:
        if projection is None:
            projection = ccrs.PlateCarree()
        # Set up the plot with a projection (e.g., PlateCarree for regular lat/lon data)
        fig, ax = plt.subplots(subplot_kw={"projection": projection})
    else:
        fig = ax.figure

    # Create cmap with white values under vmin
    cmap = _copy_cmap(cmap)
    if vmin is not None:
        cmap.set_under(color="white")

    # Label of colorbar
    if cbar_label is None:
        cbar_label = var

    cf = da.plot.contourf(
        ax=ax,
        transform=ccrs.PlateCarree(),  # Transform from data's coordinate system
        cmap=cmap,  # Choose your color map
        cbar_kwargs={"label": cbar_label},
        vmin=vmin,
        vmax=vmax,
        levels=levels,
    )

    ax.coastlines()

    if title == "time":
        if time_index is not None:
            ax.set_title(f"time = {time_years} years", pad=20)
    elif title is not None:
        ax.set_title(title, pad=20)

    return fig, ax

