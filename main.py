"""
main.py

Produces main results from paper.

In development:
    is_quality(timeseries)

"""


__author__ = "@FabriFalasca and @andrewbrettin"
__date__ = "23 February 2022"


# System
import os
import sys
from datetime import datetime

# Scientific stack
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import xarray as xr
import scipy as sc

# Computations
from scipy import linalg
from scipy.special import legendre
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
import cmocean as cmo
plt.rcParams['font.size'] = 16

# Parallelization
from joblib import Parallel, delayed

# Custom
sys.path.append('./notebooks/')
import attrs, utils

##############################################################################
## Globals
RNG = np.random.default_rng(0)
QS = np.arange(0.05, 1, 0.05)
N = 1000
BASIS = 'hermite'
BASIS_MATRIX = utils.compute_basis_matrix(basis=BASIS)
DATA_PATH = attrs.DATA_PATH
MOMENTS = ['mean', 'variance', 'skewness', 'kurtosis']
N_JOBS = 8

##############################################################################
### ----- Load and process data ----- ###
def load_filtered_tide_gauges():
    """Loads data of interest"""
    ds = xr.open_dataset(
        os.path.join(attrs.DATA_PATH, 'tide_gauges_1970-2017.nc')
    )
    return ds

def is_quality(seasonal_timeseries):
    """
    Quality checks for tide gauge.

    If there are not at least 80% of days in the record, returns false.
    """
    tmin = '1970-01-01'
    tmax = '2018-12-31'
    time_range_full = xr.DataArray(
        pd.date_range(start=tmin, end=tmax, freq='D'), dims='time'
    )
    season = seasonal_timeseries.time.dt.season.data[0]
    season_full = select_season(time_range_full, season)

    proportion = (np.sum(~np.isnan(seasonal_timeseries)) / len(season_full)).item()
    condition = (proportion > 0.8)
    return condition

def select_season(timeseries, season):
    """Selects seasons from timeseries"""
    if season not in ['DJF', 'JJA']:
        raise ValueError('Invalid season {}'.format(season))
    return timeseries.where(timeseries.time.dt.season == season, drop=True)

def days_since_1970(time):
    days = (
        (time - np.datetime64('1970-01-01')) / (86400*1e9)
    ).astype(int).data
    return days

def expand_time_coord(timeseries):
    """Expands time dimension to range to maximum timespan"""
    tmin = '1970-01-01'
    tmax = '2017-12-31'
    full_timespan = xr.DataArray(
        pd.date_range(start=tmin, end=tmax, freq='D'), dims='time'
    )
    expanded = timeseries.interp(time=full_timespan)
    return expanded

### ----- Basis functions ----- ###
def compute_basis_coeffs(slopes):
    """
    Given slopes of each quantile, returns coefficients which 
    correspond to the changes in the moments

    Parameters:
        slopes : np.array
            Vector indicating slopes of the 19 quantiles.
    Returns:
        coeffs : np.array
            Four coefficients corresponding to the changes
            in mean, variance, skewness, and kurtosis respectively.
            Units of the coefficients are the same as for the 
            quantile slopes.
    """
    coeffs = linalg.lstsq(BASIS_MATRIX, slopes)[0]
    return coeffs

def compute_weights(coeffs):
    """
    Given coefficients indicating changes in the moments,
    returns weights which indicate the relative proportion 
    of the changes in quantiles explained by that moment.

    Parameters:
        coeffs : np.array
            Coefficients indicating changes in moments.
    Returns:
        weights : np.array
            Relative weights of each change in moments.
    
    """
    weights = coeffs**2 / linalg.norm(coeffs)**2
    return weights

### ----- Quantile regression ----- ###
def compute_quantile_slopes(xs, ys):
    slopes = np.array([utils.q_regression(xs, ys, q) for q in QS])
    slopes = slopes * 365.25
    return slopes

### ----- Significance testing ----- ###
def block_shuffle_timeseries(timeseries, random_state):
    rng = np.random.default_rng(random_state)
    copy = timeseries.copy()
    groups = [arr for _, arr in copy.groupby('time.year')]
    rng.shuffle(groups)
    shuffled = xr.concat(groups, dim='time')
    return shuffled

def bootstrapped_coeffs(timeseries, N, n_jobs=0):
    """
    Bootstraps N samples from a timeseries via block shuffling and
    returns the coefficients for the moments.

    Parameters:
        timeseries : xr.DataArray
            Timeseries to bootstrap.
        N : int
            Number of bootstrap samples to use
        n_jobs : int
            If n_jobs = 0, computes coefficients without
            parallelization. Otherwise, if n_jobs is an int, uses
            joblib Parallel class to parallelize the for loop.

    Returns:
        C : np.array
            Array of shape (N, 4) of the moment coefficients for
            bootstrapped samples.
    """
    xs = days_since_1970(timeseries.time)

    # Generate seeds for random number generation with parallellization
    random_states = RNG.integers(np.iinfo(np.int32).max, size=N)

    def process_timeseries(timeseries, random_state=0):
        # Computes coefficients for block reshuffling
        shuffled_timeseries = block_shuffle_timeseries(timeseries, random_state)
        ys = shuffled_timeseries.data
        slopes = compute_quantile_slopes(xs, ys)
        coeffs = compute_basis_coeffs(slopes)
        return coeffs

    if n_jobs == 0:
        # Unparallellized
        C = np.empty((N, 4))
        for i in range(N):
            C[i, :] = process_timeseries(timeseries, random_states[i])
    elif isinstance(n_jobs, int) and n_jobs > 0:
        # Parallelized code
        C = Parallel(n_jobs=n_jobs)(
            delayed(process_timeseries)(timeseries, random_states[i]) for i in range(N)
        )
        C = np.array(C)
    else:
        raise ValueError('n_jobs is not a nonnegative integer')
    
    return C

def compute_significance(coeffs, C, alpha=0.1):
    significance_intervals = []
    for j in range(4):
        interval = pd.Interval(
            np.quantile(C[:, j], alpha/2), np.quantile(C[:, j], 1-alpha/2)
        )
        significance_intervals.append(interval)
    sigs = np.empty(4, dtype=bool)
    for j in range(4):
        sigs[j] = coeffs[j] not in significance_intervals[j]
    return sigs

### ----- Saving ----- ###
def save_output(gauges_list, coeffs_list, weights_list, sigs_list, season,
        path=os.path.join(DATA_PATH, 'computation')):
    for l in [gauges_list, coeffs_list, weights_list, sigs_list]:
        l = np.array(l)
    np.save(os.path.join(path, '{}_gauges_{}.npy'.format(season, BASIS)), gauges_list)
    np.save(os.path.join(path, '{}_coeffs_{}.npy'.format(season, BASIS)), coeffs_list)
    np.save(os.path.join(path, '{}_weights_{}.npy'.format(season, BASIS)), weights_list)
    np.save(os.path.join(path, '{}_sigs_{}.npy'.format(season, BASIS)), sigs_list)

### ----- Plotting ----- ###
def plot_tide_gauge_coeffs(m, lats, lons, coeffs_list, sigs_list, season,
        ax=None, **kwargs):
    """
    Plots coefficients of tide gauges for a specific moment.

    Parameters:
        m : int
            Number from 0 to 3 indicating which moment to plot.
        lats : array-like
            Latitudes of tide gauges.
        lons : array-like
            Longitudes of tide gauges.
        coeff_list : np.array
            Array of shape (L, 4) giving the coefficients of the 
            moments for all of the tide gauges, where L is the number
            of tide gauges plotted.
        sigs_list : np.array
            Array of shape(L, 4) giving the statistical significances 
            of the changes in moments for all the tide gauges.
        season : string
            'DJF' or 'JJA' indicating NH winter or summer
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            Plotting axis.
    Returns:
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            Plotting axis.
    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(15,8),
            subplot_kw={'projection' : ccrs.PlateCarree(central_longitude=-90)}
        )
    else:
        fig = plt.gcf()
    
    # Set plot properties
    vmax = 5.0 if m==0 else 2.0
    vmin = -vmax
    moment = MOMENTS[m]
    markers = ['o' if sig else 'X' for sig in sigs_list[:, m]]

    # Plot
    for lat, lon, color, marker in zip(lats, lons, coeffs_list[:, m], markers):
        cax = ax.scatter(
            lon, lat,
            c=color, cmap=cmo.cm.balance, vmin=vmin, vmax=vmax,
            marker=marker, s=54, edgecolor='k',
            transform=ccrs.PlateCarree(),
            **kwargs
        )
    
    # Plot features
    ax.set_global()
    ax.add_feature(cartopy.feature.LAND, color=(0.7,0.7,0.7))
    gl = ax.gridlines(draw_labels=True)
    gl.xlocator = mticker.FixedLocator(np.arange(-270, 90, 90))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 45))
    gl.top_labels = False

    # Colorbar
    cb = plt.colorbar(cax, ax=ax, fraction=0.024, pad=0.022)
    cb.set_label(r'mm year$^{-1}$')

    # Axis properties
    ax.set(
        title='{} changes in {}'.format(season, moment),
        xlabel='Longitude',
        ylabel='Latitude'
    )

    fig.tight_layout()

    return ax

def plot_tide_gauge_weights(m, lats, lons, weights_list, season,
        ax=None, **kwargs):
    """
    Plots the weights of the changes in moments.

    Parameters:
        m : int
            Number from 0 to 3 indicating which moment to plot.
        lats : array-like
            Latitudes of tide gauges.
        lons : array-like
            Longitudes of tide gauges.
        weights_list : np.array
            Array of shape (L, 4) giving the weights of the 
            moments for all of the tide gauges, where L is the number
            of tide gauges plotted.
        season : string
            'DJF' or 'JJA' indicating NH winter or summer
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            Plotting axis.
    Returns:
        ax : cartopy.mpl.geoaxes.GeoAxesSubplot
            Plotting axis.
    """
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(15,8),
            subplot_kw={'projection' : ccrs.PlateCarree(central_longitude=-90)}
        )
    else:
        fig = plt.gcf()
    
    # Set plot properties
    moment = MOMENTS[m]

    # Plot
    cax = ax.scatter(
        lons, lats, c=weights_list[:,m], cmap=cmo.cm.amp, vmin=0, vmax=1,
        marker='o', s=54, edgecolor='k',
        transform=ccrs.PlateCarree(),
        **kwargs
    )

    # Plot features
    ax.set_global()
    ax.add_feature(cartopy.feature.LAND, color=(0.7,0.7,0.7))
    gl = ax.gridlines(draw_labels=True)
    gl.xlocator = mticker.FixedLocator(np.arange(-270, 90, 90))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 45))
    gl.top_labels = False

    # Colorbar
    cb = plt.colorbar(cax, ax=ax, fraction=0.024, pad=0.022)
    cb.set_label('Weight')

    # Axis properties
    ax.set(
        title='{} weight of {} changes'.format(season, moment),
        xlabel='Longitude',
        ylabel='Latitude'
    )

    fig.tight_layout()

    return ax

##############################################################################

def main():
    # Load tide gauges
    ds = load_filtered_tide_gauges()

    for season in ['DJF', 'JJA']:
        # List of variables
        gauges_list = []
        coeffs_list = []
        sigs_list = []
        weights_list = []

        for station_name, gauge in ds.items():
            # Select season and process timeseries
            timeseries = select_season(gauge, season)
            if not is_quality(timeseries):
                # Ignore timeseries
                continue
            timeseries = expand_time_coord(timeseries)

            # Quantile regression
            xs = days_since_1970(timeseries.time)
            ys = timeseries.data
            slopes = compute_quantile_slopes(xs, ys)

            # Projection onto basis functions
            coeffs = compute_basis_coeffs(slopes)
            weights = compute_weights(coeffs)

            # Bootstrap significance tests
            C = bootstrapped_coeffs(timeseries, N, n_jobs=N_JOBS)
            sigs = compute_significance(coeffs, C)
        
            # Append values
            gauges_list.append(station_name)
            coeffs_list.append(coeffs)
            weights_list.append(weights)
            sigs_list.append(sigs)
        
        # Save values
        save_output(gauges_list, coeffs_list, weights_list, sigs_list, season)

        # Plotting
        lats = [ds[station].latitude for station in gauges_list]
        lons = [ds[station].longitude for station in gauges_list]
        for i, moment in enumerate(MOMENTS):
            # Coefficients
            fig, ax = plt.subplots(
                figsize=(15,8),
                subplot_kw={'projection' : ccrs.PlateCarree(central_longitude=-90)}
            )
            plot_tide_gauge_coeffs(
                i, lats, lons, coeffs_list, sigs_list, season,
                ax=ax, alpha=0.8
            )
            path = os.path.join(attrs.VISUALIZATION_PATH)
            fname = '{}_changes_{}.png'.format(season, moment)
            fig.savefig(os.path.join(path, fname), dpi=300)

            # Weights
            fig, ax = plt.subplots(
                figsize=(15,8),
                subplot_kw={'projection' : ccrs.PlateCarree(central_longitude=-90)}
            )
            plot_tide_gauge_weights(
                i, lats, lons, weights_list, season,
                ax=ax, alpha=0.8
            )
            path = os.path.join(attrs.VISUALIZATION_PATH)
            fname = '{}_weights_{}_{}.png'.format(season, moment, BASIS)
            fig.savefig(os.path.join(path, fname), dpi=300)

    return 0


if __name__ == "__main__":
    main()
