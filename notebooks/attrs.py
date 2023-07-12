"""
attrs.py

Module for various Python attributes to keep track of.
"""

__author__ = '@andrewbrettin'


import numpy as np
import pandas as pd
import xarray as xr
import scipy as sc

from scipy import linalg
from scipy import stats
from scipy import special
from scipy.special import legendre

import statsmodels.api as sm
from patsy import dmatrices

import os
# PATH = '/Users/andrewbrettin/Tomfoolery/ocean_science/'    # Modify to user
PATH = '/Users/andrewbrettin/Research/'
DATA_PATH = os.path.join(PATH, 'quantreg/data/')
VISUALIZATION_PATH = os.path.join(PATH, 'quantreg/visualization/')


##### Data #####
tmin = '1846-01-04T12:00:00'    # Starting time for tide gauge selection
tmax = '2019-07-29T12:00:00'    # Ending time for tide gauge selection
# # tide_gauges = np.load()
# tide_gauges_dict = np.load(
#     os.path.join(DATA_PATH, 'daily_tide_gauges_dict.npy'), allow_pickle=True
# ).item()

##### quantiles #####
qs = np.arange(0.05, 1.00, 0.05)
qs_fine = np.arange(0.001, 1.000, 0.001)


##### Distributions #####

# Need to define distribution for final kurtosis basis function
class pearson7_gen(stats.rv_continuous):
    "Pearson distribution"
    def _pdf(self, x, shape):
        return (1 / (special.beta(shape - 0.5, 0.5))
            * (1 + x**2)**-shape
        )

pearson7 = pearson7_gen()

start_dist_list = [stats.norm, stats.norm, stats.pearson3, stats.beta]
end_dist_list = [stats.norm, stats.norm, stats.pearson3, pearson7]

start_args = [
    {'loc' : 0, 'scale':1},
    {'scale' : 1},
    {'skew':-0.5, 'loc': 0, 'scale':1},
    {'a': 1.5, 'b':1.5, 'loc':-2, 'scale': 4}
]
end_args = [
    {'loc' : 1, 'scale':1},
    {'scale' : 1.5},
    {'skew':0.5, 'loc': 0, 'scale':1},
    {'shape':2, 'loc': 0, 'scale': 1}
]

start_quantiles = np.sort([
    start_dist.ppf(qs, **start_arg) 
    for start_dist, start_arg in zip(start_dist_list, start_args)
])

end_quantiles = np.sort([
    end_dist.ppf(qs, **end_arg) 
    for end_dist, end_arg in zip(end_dist_list, end_args)
])

quantile_changes = [
    end_quantile - start_quantile 
    for start_quantile, end_quantile in zip(start_quantiles, end_quantiles)
]

##### Basis functions #####

moments = ['mean', 'var', 'skew', 'kurtosis']
moment_basis_functions = dict(zip(moments, quantile_changes))

# Legendre basis functions
legendre_basis_functions = []
for i in range(4):
    moment_basis  = list(moment_basis_functions.values())[i]
    unscaled_legendre_basis = special.legendre(i)(2*qs - 1)   # Set interval to [0,1]
    scale = linalg.norm(moment_basis) / linalg.norm(unscaled_legendre_basis)
    legendre_basis_functions.append(scale * special.legendre(i)(2*qs - 1))
legendre_basis_functions = dict(zip(moments, legendre_basis_functions))

# Hermite basis functions
normal_quantiles = stats.norm.ppf(qs, loc=0, scale=1)
hermite_basis_functions = [
    np.ones_like(qs),
    normal_quantiles / 2,
    (normal_quantiles**2 - 1) / 6,
    (normal_quantiles**3 - 3 * normal_quantiles) / 24
]
hermite_basis_functions = dict(zip(moments, hermite_basis_functions))

