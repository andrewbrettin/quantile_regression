# Libraries and stuff

import numpy as np
import numpy.ma as ma

# Remove warning and live a happy life
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd

'''

MAIN CODE for quantifying (linear) changes in distributions from time series
Note: here we consider seasonal time series (focusing on JJA, DJF etc.). Few small changes in the block-bootstrapping functions 
are needed if we consider the full time series.

Contacts:

Fabri Falasca; fabri.falasca@nyu.edu
Andrew Brettin; aeb783@nyu.edu

'''

################## What's in here

# (a) Code for quantile regression + statistical significance

'''
Functions

- q_regression: 
for a given time series and quantile q it estimates the quantile regression

- q_regression_significance: 
for a given time series and quantile q it estimates the quantile regression 
and its statistical significance

- q_reg_analysis:
compute the quantile regression and significance for a set of quantiles q in [0,1]; 
this one is the main function for the quantile regression step


'''

# (b) Code quantifying changes in the first four statistical moments

'''

- basis_functions:
the 4 basis functions derived from the Cornish-Fisher expansion

- bootstrap_time_series:
it block-bootstrapps a time series n time and returns the result

- changes_in_moments:
for a given time series it computes the slopes in mean, variance, skewness and kurtosis of 
the distribution plus their significance;
this one is the main function for quantifying trends in moments from quantiles

'''

################# Step (a): QUANTILE REGRESSION

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.utils import resample

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

    df = pd.DataFrame({'days': x_not_nans, 'sea_level': y_not_nans})
    model_q = smf.quantreg('sea_level ~ days', df).fit(q=q,max_iter = 1000,p_tol = 1e-3)

    # regression line
    get_y = lambda a, b: a * x + b

    # y_predicted
    y_predicted = get_y(model_q.params['days'],model_q.params['Intercept'])
    
    # slope
    slope = model_q.params['days']
    
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
    
    # Indices to split

    # Example:
    # np.split(x,[a,b,c]) will split x 4 parts:
    # [x < a; a <= x < b; b <= x <c; x >=c

    split_idx = []
    for i in range(len(diff)):
        if diff[i] != 1.:
            split_idx.append(i+1) # Need to add 1

    # Here the blocks
    blocks = np.array(np.split(detrended,split_idx))
    
    # We estimate a PDF of slopes to compute the significance
    s_prime = []

    #print('Bootstrap test for statical significance started')

    for i in range(n):

        # Blockboostrap + resampling the blocks
        block_bootstrapped = resample(blocks,replace = True).flatten()
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

################# Quantile regression analysis for a time series

def q_reg_analysis(xt,yt,qs,n):

    # initialize arrays
    y_predicted=[]  # trends
    slopes=[]       # slopes
    significance=[]  # significance (True ---> it is significance)
    upper_bounds=[]  # upper level bound (90% confidence)
    lower_bounds=[]  # lower level bound (90% confidence)

    for q in qs:

        q=np.round(q, 2)

        y_predicted_q, slope_q, significance_q, upper_bound_q, lower_bound_q = q_regression_significance(xt, yt, q, n)

        y_predicted.append(y_predicted_q)
        slopes.append(slope_q)
        significance.append(significance_q)
        upper_bounds.append(upper_bound_q)
        lower_bounds.append(lower_bound_q)
        
        print('Analysis for quantile '+str(q)+' complete')

    y_predicted=np.array(y_predicted)
    slopes=np.array(slopes)
    significance=np.array(significance)
    upper_bounds=np.array(upper_bounds)
    lower_bounds=np.array(lower_bounds)
    
    return y_predicted, slopes, significance, upper_bounds, lower_bounds

################# Step (b): PROJECTION ONTO BASIS

from scipy.stats import norm
from scipy import linalg
import pandas as pd

# Defining basis

def basis_functions(qs):
    
    # input: array of qs. E.g., qs = np.arange(0.02, 1.00, 0.02)
    # the four basis functions
    b1 = np.array([1] * len(qs))
    b2 = (1/2)*norm.ppf(qs, loc=0, scale=1)
    b3 = (1/6)*( (norm.ppf(qs, loc=0, scale=1)**2 ) - 1)
    b4 = (1/24)*( (norm.ppf(qs, loc=0, scale=1)**3 ) - 3*norm.ppf(qs, loc=0, scale=1))
    
    return np.array([b1,b2,b3,b4])

# Block bootstrap a time series (trends included)

from sklearn.utils import resample

def bootstrap_time_series(xt,yt,n):

    # Input:
    #       - xt: is time
    #       - yt: is the variable of interest (e.g., sea level)
    #       - n:  number of blockboostrapps

    # To capture a season-length of data (blocks)

    # (a) look at the difference between 1 time step and another.
    # (b) Our delta t is 1 day
    #     - if the diff is 1, then we are in the same season;
    #     - if the diff is more than 1 ---> we just shifted to another reason.

    xt = xt - xt[0] # Not that it changes anything...
    diff = np.diff(xt)

    # Indices to split

    # Example:
    # np.split(x,[a,b,c]) will split x 4 parts:
    # [x < a; a <= x < b; b <= x <c; x >=c

    split_idx = []
    for i in range(len(diff)):
        if diff[i] != 1.:
            split_idx.append(i+1) # Need to add 1

    # Here the blocks
    blocks = np.array(np.split(yt,split_idx))

    # Bootstrapping

    bootstrapped_yt = []

    for i in range(n):

        # Blockboostrap + resampling the blocks
        block_bootstrapped = resample(blocks,replace = True).flatten()
        bootstrapped_yt.append(block_bootstrapped)

    bootstrapped_yt = np.array(bootstrapped_yt)

    return bootstrapped_yt

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
    
    bootstrapped_ts =  bootstrap_time_series(xt,yt,n)

    B = np.empty((n, len(qs)))    # Initialize matrix

    # Populate quantile regression matrix
    for i in range(n):
        for j, q in enumerate(qs):
            ys = bootstrapped_ts[i, :]
            # B size is n (bootstrapped samples) by len(qs) quantile trends
            B[i, j] = q_regression(xt, ys, q)[1]     
            
     # From bootstrapped matrix, compute coefficient matrix using the basis functions
    C = np.zeros((n, 4))
    
    basis = basis_functions(qs)

    for i in range(n):
        C[i, :] = linalg.lstsq(np.transpose(basis), B[i, :])[0] 

    # Create 90% significance intervals:
    # significance_intervals[i] is the significance interval for the ith moment coefficient.

    significance_intervals = [
        pd.Interval(np.quantile(C[:, j], 0.05), np.quantile(C[:, j], 0.95),)
        for j in range(4)
    ]         
    
    # Quantile regression of the original time series
    slopes = []
    for q in qs:

        q=np.round(q, 2)
        slope = q_regression(xt,yt,q)[1]
        slopes.append(slope)
    slopes = np.array(slopes)
    
    # Compute also the coefficients for the main quantile regressions
    coeffs = linalg.lstsq(np.transpose(basis), slopes)[0]
    # Compute significances
    sigs = [coeffs[i] not in significance_intervals[i] for i in range(4)]
    
    return coeffs, sigs
