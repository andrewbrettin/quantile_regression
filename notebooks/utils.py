"""
utils.py

Module for miscellaneous utilities
"""

__author__ = "@andrewbrettin and @FabriFalasca"
__date__ = "24 February 2022"

import numpy as np
import pandas as pd
import xarray as xr
import scipy as sc

from scipy import linalg
from scipy.stats import rv_continuous, norm, pearson3, beta
from scipy.special import legendre

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

import matplotlib.pyplot as plt

import attrs
import os
import re


### FUNCTIONS ###
### --------- Loading data --------- ###
def load_tide_gauges_dict():
    d = np.load(
        os.path.join(attrs.DATA_PATH, 'daily_tide_gauges_dict.npy'), allow_pickle=True
    ).item()
    return d

### --------- Basis functions --------- ###
def start_quantiles(qs):
    # Starting quantiles for the moment basis distributions
    return np.sort([
        start_dist.ppf(qs, **start_arg) 
        for start_dist, start_arg in zip(attrs.start_dist_list, attrs.start_args)
    ])

def end_quantiles(qs):
    # Ending quantiles for the moment basis distributions
    return np.sort([
        end_dist.ppf(qs, **end_arg) 
        for end_dist, end_arg in zip(attrs.end_dist_list, attrs.end_args)
    ])

def moment_basis(qs):
    # Given a list of quantiles qs (e.g., np.arange(0.05,1,0.05)),
    # returns a list of quantile changes for each moment basis
    return np.array([
        end_quantile - start_quantile 
        for start_quantile, end_quantile in zip(start_quantiles(qs), end_quantiles(qs))
    ])

def legendre_scaling_coefficients(qs):
    scales = np.zeros(4)
    for i in range(4):
        unscaled_legendre_basis = legendre(i)(2*qs - 1)   # Set interval to [0,1]
        scales[i] = linalg.norm(moment_basis(qs)[i]) / linalg.norm(unscaled_legendre_basis)
    return scales
    
def legendre_basis(qs, scaled=True):
    # Returns a list of legendre basis functions
    basis_functions = []
    for i in range(4):
        unscaled_legendre_basis = legendre(i)(2*qs - 1)   # Set interval to [0,1]
        if scaled:
            scale = linalg.norm(moment_basis(qs)[i]) / linalg.norm(unscaled_legendre_basis)
            basis_functions.append(scale * legendre(i)(2*qs - 1))
        else:
            basis_functions.append(unscaled_legendre_basis)
    return np.array(basis_functions)

def compute_basis_matrix(basis='legendre'):
    # Given a basis, returns a coefficient matrix
    qs = attrs.qs
    if basis == 'moments':
        basis_funcs = list(attrs.moment_basis_functions.values())
    elif basis == 'legendre':
        basis_funcs = list(attrs.legendre_basis_functions.values())
    elif basis == 'hermite':
        basis_funcs = list(attrs.hermite_basis_functions.values())
    else:
        raise ValueError(
            "Basis must be either 'moments' or 'legendre' or 'hermite'"
        )
    A = np.zeros((len(qs), 4))
    for k in range(4):
        A[:, k] = basis_funcs[k]
    return A


### --------- Quantile regression --------- ###
def q_regression(xt,yt,q):
    """
    Performs quantile regression.
    author: @FabriFalasca
    
    Input: 
        xt : time
        yt : variable of interest
        q : quantile of interest
    Returns:
        trend
        slope
    """
    # Set the time vector going from 0 to T
    xt = xt - xt[0]

    x = xt.copy()
    y = yt.copy()
    # Do not consider nan
    y_not_nans = y[~np.isnan(y)]
    x_not_nans = x[~np.isnan(y)]

    df = pd.DataFrame({'days': x_not_nans, 'sea_level': y_not_nans})
    model_q = smf.quantreg('sea_level ~ days', df).fit(q=q, max_iter=1000, p_tol=1e-3)

    # slope
    slope = model_q.params['days']

    return slope

### --------- Miscellaneous --------- ###
def string_regex(string):
    """Given a string, returns a version without spaces, 
    special characters or uppercase letters."""
    return re.sub('[^A-Za-z0-9 ]+', '', string).replace(' ', '_').lower()