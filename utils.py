"""
A set of utlity functions used throughout the ivf_embryo_prediction.

ivf_embryo_prediction, Machine-learnt models for predicting chance
of suitable embryo for D5 transfer or freezing.

© University of Southampton, IT Innovation Centre, 2020

Copyright in this software belongs to University of Southampt
University Road, Southampton, SO17 1BJ, UK.

This software may not be used, sold, licensed, transferred, copied
or reproduced in whole or in part in any manner or form or in or
on any media by any person other than in accordance with the terms
of the Licence Agreement supplied with the software, or otherwise
without the prior written consent of the copyright owners.

This software is distributed WITHOUT ANY WARRANTY, without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE, except where stated in the Licence Agreement supplied with
the software.

Created Date :          24-06-2020

Created for Project :   Fertility predict

Author: Francis P. Chmiel

Email: F.P.Chmiel@soton.ac.uk
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

def remove_axes(ax):
    """
    Removes the top and right axis from a matplotlib axis object.

    Parameters:
    ax, matplotlib.Axes
        The axis object to remove the axis from.
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def change_ticks_fontsize(ax, size=7):
    """
    Changes the x and y tick labels fontsize.

    Parameters:
    -----------
    ax , matplotlib.Axes
        The axis object to change the fontsize of.
    """
    plt.setp(ax.get_xticklabels(), fontsize=size)
    plt.setp(ax.get_yticklabels(), fontsize=size)

def confidence_intervals(scores, alpha=0.95):
    """
    Calculates the confidence intervals for a list of model performances 
    (scores). The scores should be the result of a bootstrap modelling 
    technique.

    Parameters:
    -----------
    scores : array-like
        The bootstrapped samples of each models performance.

    alpha : float
        The size of the confidence interval to compute. Must be between 0 and 1.

    Returns:
    --------
    confidence : tuple
        Returns the lower and upper bound of the confidence intervals.
    """
    upper_percentile = ((1.0-alpha)/2.0) * 100
    lower = np.percentile(scores, upper_percentile)
    lower_percentile = (alpha+((1.0-alpha)/2.0)) * 100
    upper = np.percentile(scores, lower_percentile)
    return lower, upper

def plot_line_with_confidence(x, 
                              y, 
                              err, 
                              ax=None, 
                              plot_params={}, 
                              fill_params={},
                              xlims=[0,25],
                              ylims=[0,100],
                              xlabel=None,
                              ylabel=None):
    """
    Plots a line with shaded confidence intervals.
    
    Parameters:
    -----------
    x : np.array, 
        The x data
        
    y : np.array, 
        The y data.
        
    err : np.array, 
        Error on the y-axis values. This assumes a symmetric error 
        so bounds of the errors are equal to (y-err, y+err). 
        
    ax : matplotlib.Axes (default=None),
        The axis object data is plotted too.
    
    plot_params : dict (default={}),
        Parameters passed to the plt.plot call.
    
    fill_params : dict (default={}),
        Parameters passed to the plt.fill_between call.
    
    xlims : list (default=[0,25]),
        Limits of the x-axis.
        
    ylims : list (default=[0,100]),
        Limits of the y-axis.
    
    xlabel : str (default=None),
        The x-axis label.
        
    ylabel : str (default=None),
        The y-axis label. 
        
    Returns:
    --------
    ax : matplotlib.Axes (default=None),
        The axes instance the data was plotted too.
    """
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.plot(x, y, **plot_params)
    ax.fill_between(x, y-err, y+err, **fill_params)

    # format axis, add labels etc
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
                              
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
                              
    remove_axes(ax)
                              
    return ax
                              
def calculate_proportion_error(ps, counts, z=1.96):
    """
    Calculates the standard error on a proportion.
    
    Parameters:
    -----------
    ps : np.array,
        An array of proportions (i.e., each value is between 0 and 1 and represents
        the ratio of two classes). A single element is the mean of a binary varible.
    
    counts : np.array,
        The total number of values when calculating each proportion (each element of ps).
    
    z : float (default=1.96)
        Z value for confidence interval. z = 1.96 corresponds to a 95% confidence interval.
    
    Returns:
    -------
    std_err : np.array,
        The standard error on each calculated proportions.
    """
    return z * ((ps * (1-ps)) / counts)**0.5 