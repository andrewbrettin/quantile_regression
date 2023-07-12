# Libraries and stuff

import numpy as np
import numpy.ma as ma

# Remove warning and live a happy life
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd

'''

Tests on synthetic data.
For a given time series we compute the quantile regression and condense results
using the basis functions derived from the Cornish-Fisher Expansion.

Note: the statistical test used here is bootstrapping and not block-bootstrapping as
in the main paper. The reason is simple: our time series is created by randomly
sample at each time step from a drifting distribution. Therefore, point by point are
identically independently distributed (iid) processes by construction. This motivates
the use of a simple bootstrapping procedure.

Contacts:

Fabri Falasca; fabri.falasca@nyu.edu

'''

from scipy.stats import norm
from scipy import linalg
import pandas as pd
from sklearn.utils import resample
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Defining basis

def basis_functions(qs):

    # input: array of qs. E.g., qs = np.arange(0.02, 1.00, 0.02)
    # the four basis functions
    b1 = np.array([1] * len(qs))
    b2 = (1/2)*norm.ppf(qs, loc=0, scale=1)
    b3 = (1/6)*( (norm.ppf(qs, loc=0, scale=1)**2 ) - 1)
    b4 = (1/24)*( (norm.ppf(qs, loc=0, scale=1)**3 ) - 3*norm.ppf(qs, loc=0, scale=1))

    return np.array([b1,b2,b3,b4])

def q_regression(xt,yt,q):

    # Input:
    #       - xt: is time
    #       - yt: is the variable of interest (sea level for us)
    #       - q:  quantile of interest (i.e., q = 0.05, 0.5, 0.95)

    # We have nans in the data: we do not consider them in the analysis

    # Output:
    #       - trend
    #       - slope

    # Set the time vector going from 0 to T
    xt = xt - xt[0]

    x = xt.copy()
    y = yt.copy()
    # Do not consider nan
    y_not_nans = y[~np.isnan(y)]
    x_not_nans = x[~np.isnan(y)]

    df = pd.DataFrame({'time': x_not_nans, 'timeseries': y_not_nans})
    model_q = smf.quantreg('timeseries ~ time', df).fit(q=q,max_iter = 2000,p_tol = 1e-3)

    # regression line
    get_y = lambda a, b: a * x + b

    # y_predicted
    y_predicted = get_y(model_q.params['time'],model_q.params['Intercept'])

    # slope
    slope = model_q.params['time']

    return y_predicted, slope


def q_regression_significance(xt,yt,q,n):

    # Input:
    #       - xt: is time
    #       - yt: is the variable of interest (sea level for us)
    #       - q:  quantile of interest (i.e., q = 0.05, 0.5, 0.95)
    #       - n:  number of blockboostrapps
    #       - method: 'sklearn' or 'statsmodels'

    # We have nans in the data: we do not consider them in the analysis

    # Output:
    #       - y_predicted:                     trend line
    #       - slope:                           slope of the trend
    #       - significance:                   'Significant' or 'Not Significant'
    #       - s_prime_confidence_UPPER_BOUND:  upper bound
    #       - s_prime_confidence_LOWER_BOUND:  lower bound

    y_predicted, slope = q_regression(xt,yt,q)

    # Detrend the data
    detrended = yt-y_predicted

    # To capture a season-length of data (blocks)

    # (a) look at the difference between 1 time step and another.
    # (b) Our delta t is 1 day
    #     - if the diff is 1, then we are in the same season;
    #     - if the diff is more than 1 ---> we just shifted to another reason.
    xt = xt - xt[0] # Not that it changes anything...
    diff = np.diff(xt)

    # We estimate a PDF of slopes to compute the significance
    s_prime = []

    #print('Bootstrap test for statical significance started')

    for i in range(n):

        # Blockboostrap (process is iid, so no need for block-bootstrap here)
        block_bootstrapped = resample(yt,replace = True).flatten()
        # We add the originally estimated trend back again
        block_bootstrapped_plus_s = block_bootstrapped + y_predicted

        y_predicted_prime, slope_prime = q_regression(xt,block_bootstrapped_plus_s,q)

        s_prime.append(slope_prime)


        #if(i%100 == 0):
        #    print('Progress: '+str(np.round((i+1)/n,2)));
        #print('Iteration '+str(i+1)+'/'+str(n)+' completed')

    s_prime = np.array(s_prime)

    # We say that it is significant IF 95% of the bootstrapped values has the same sign (here +)
    # np.sign(x) == +1 if x > 0, 0 if x == 0 and -1 if x < 0
    sign_s = np.sign(slope)
    sign_s_prime = np.sign(s_prime)

    # Saving the ratio of positive slopes
    ratio_plus = np.sum(sign_s_prime == 1.)/len(sign_s_prime)
    # Saving the ratio of negative slopes
    ratio_negative = np.sum(sign_s_prime == -1.)/len(sign_s_prime)
    # Saving the ratio of slopes == zero
    ratio_zero = np.sum(sign_s_prime == 0)/len(sign_s_prime)

    # The (original) slope is significant IF at least 95% of the bootstrapped s_prime have the same sign
    if sign_s == 1:
        if ratio_plus >= 0.95:
            significance = 'True'
        else:
            significance = 'False'
    elif sign_s == 0:
        if ratio_zero >= 0.95:
            significance = 'True'
        else:
            significance = 'False'
    elif sign_s == -1:
        if ratio_negative >= 0.95:
            significance = 'True'
        else:
            significance = 'False'

    # 90% confidence bounds
    upper_bound = np.percentile(s_prime,95,axis=0)
    lower_bound = np.percentile(s_prime,5,axis=0)

    return y_predicted, slope, significance, upper_bound, lower_bound

# Bootstrapping

def bootstrap(ts,n):

    # input:
    # - time series ts
    # - n: number of bootsrap

    # output:
    # - n bootstrapped time series

    bootstrapped_ts = []

    for i in range(n):

        bootstrapped_ts.append(resample(ts,replace = True).flatten())

    bootstrapped_ts = np.array(bootstrapped_ts)

    return bootstrapped_ts

def changes_in_moments(xt,yt,n,qs):

    # Input:
    # xt: time array
    # yt: time series
    # n: number of bootstrap samples
    # qs: quantiles

    # Output:
    # coeffs. 4 numbers: each one tells you the slope in [mean, variance, skewness, kurtosis]
    # sigs. 4 numbers: each one tells you if the slope in [mean, variance, skewness, kurtosis]
    # is significant or not

    # Important, if the time series has just nans,
    # then we return coeff = np.array([np.nan,...]) and sigs = np.array([np.nan,...])

    #####################################################################################################
    ########### Quantifying linear changes in percentiles driven by linear changes in moments ###########
    #####################################################################################################

    print('Computing quantile regression for original time series')

    ################## Step (a)

    # Linear quantile regression

    slopes = []
    trends = []
    for q in qs:

        q=np.round(q, 3)
        #print('qr for q = '+str(q))
        qr = q_regression(xt,yt,q)

        trends.append(qr[0])

        slopes.append(qr[1])

    trends = np.array(trends)
    slopes = np.array(slopes)

    ################## Step (b)

    print('Computing projection onto basis for original time series')

    # Projection onto basis functions
    basis = basis_functions(qs)
    coeffs = linalg.lstsq(np.transpose(basis), slopes)[0]

    #####################################################################################################
    ########### Quantifying the statistical significance for each slope #################################
    #####################################################################################################

    # We compute the projection onto basis functions for a large set of bootstrapped time series
    # Statistical significance is computed relative to the bootstrapped histogram

    print('Statistical significance started')

    # bootstrapped time series
    bootstrapped_ts =  bootstrap(yt,n)

    B = np.empty((n, len(qs)))    # Initialize matrix

    print('Quantile regression on the bootstrapped time series')

    # Populate quantile regression matrix
    for i in range(n):
        for j, q in enumerate(qs):
            ys = bootstrapped_ts[i, :]
            # B size is n (bootstrapped samples) by len(qs) quantile trends
            B[i, j] = q_regression(xt, ys, q)[1]

    # From bootstrapped matrix, compute coefficient matrix using the basis functions
    C = np.zeros((n, 4))

    print('Projection onto basis for the bootstrapped time series')

    for i in range(n):
        C[i, :] = linalg.lstsq(np.transpose(basis), B[i, :])[0]

    # Create 90% significance intervals
    # significance_intervals[i] is the significance interval for the ith moment coefficient.

    significance_intervals = [
            pd.Interval(np.quantile(C[:, j], 0.05), np.quantile(C[:, j], 0.95),)
            for j in range(4)
        ]

    # Compute significances
    sigs = [coeffs[i] not in significance_intervals[i] for i in range(4)]

    print('Done')

    return coeffs, sigs, trends, slopes
