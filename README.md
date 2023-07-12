# Quantile regression

Check out our paper in Environmental Data Science [Exploring the nonstationarity of coastal sea level probability distributions](https://doi.org/10.1017/eds.2023.10)


We focus on daily-resolved, seasonal Tide Gauges records downloaded from the University of Hawaii Sea Level Center (https://uhslc.soest.hawaii.edu/) and quantify changes in the full sea level distribution. We do this in two steps: 

(a) we apply quantile regression to the observed sea level record. Unlike ordinary least squares, quantifying changes in the conditional mean over time, quantile regression allows for assessing shifts in all percentiles. The significance of each quantiles' slope is quantified by block-bootstrapping (i.e., block size of one season). 

(b) Trends in quantiles allow to quantify (linear) changes in the distribution. However, information of N quantile slopes for each time series is difficult to digest and interpret. We then project trends in quantiles onto a set of four, orthogonal basis functions linked to the first statistical moments: mean, variance, skewness and kurtosis. To do so, we first derive an analytical relationship linking changes in quantiles to changes in the moments of the distribution. Such derivation is based on the Cornish-Fisher expansion (https://en.wikipedia.org/wiki/Cornish%E2%80%93Fisher_expansion) first proposed in 1937. This provides a general, exact way to relate changes in moments to changes in percentiles.

The derivation of such basis can also be found in this page in the document "moments_and_basis/Moments_and_quantiles.pdf".

The study was inspired by this analysis: https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1002/2016JD025292

## License

To do

## Instructions
* Modify `notebooks/attrs` so that the variable `PATH` points to your working directory, where `quantreg` is cloned.

## File overview

Directory contents ordered by priority

```
quantreg/
  README.md                         # You are here
  environment.yml                   # Environment file for this project
  notebooks/
    tide_gauge_download.ipynb       # Downloads the tide gauge data using ERDDAP
    fitting_pearson_dists.ipynb     # Creates the basis distributions for kurtosis
    percentile_trends.ipynb         # Applies basis functions/quantile regression
                                    #   to idealized data
    basis_distributions.ipynb       # Plots basis distributions
    exploration/		    # Various exploratory notebooks
      statsmodels.ipynb	              # Explore statsmodels API
      query_dictionary.ipynb          # Shows how to subset values from tide_gauges_dict
      tide_gauge_quantreg.ipynb       # Example regression on some tide gauges
      assess_coeff_significance...    # Shows how to compute significances
      pensacola_bootstrap.ipynb       # Assesses significance for coeffs at Pensacola
    utils.py                        # Various utility functions
    attrs.py                        # Various attributes, such as ds dictionary
  visualization/
    tide_gauge_timeseries/          # Plots of TG observed sea level
  data/
    tide_gauge_timeseries/          # Individual netcdf files for each timeseries
    tide_gauge_table.csv            # Table of fetched tide gauge data available from 
    daily_tide_gauges_dict.npy      # Dictionary of daily tide gauge datasets created in
                                    #   tide_gauge_download.ipynb
    bootstrapped_pensacola_JJASO... # Bootstrapped pensacola timeseries for computing
                                    #   statistical significance in basis coefficients
    timevector_JJASON.npy           # Time data for bootstrapped_pensacola_JJASON.npy
    pensacola_bootstrap_slopes.npy  # Bootstrapped pensacola timeseries for computing
                                    #   statistical significance in quantile slopes
    slopes_Pensacola_JJASON/        # Slopes and trendlines for quantiles 0.01-0.99
                                    #   for quantile slope significance test
```

