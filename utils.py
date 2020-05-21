"""
A set of utlity functions used throughout the IVF project.

F. P. Chmiel, IT Innovation centre 2020
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

def objects_to_int(x):
    """
    Converts all elements of x to an integer. 

    Parameters:
    -----------
    x, Iterable{float,str,int} or float
        An iterable containing numeric values or a float.

    Returns:
    --------
    x, list{int, NaN}
        The converted iterable or float.
    """
    if type(x)==float:
        try:
            x = [int(x)]
        except:
            x = []
    else:
        try:
            x = [int(i) for i in x]
        except:
            x = []
            warnings.warn('x is neither an iterable or float, returning NaN')
    return x

def follicle_summary(follicles, cutoff=10):
    """
    Given a list of follicles diameters return the mean diameter, number of 
    follicles and the percent above the cutoff diameter.

    Parameters:
    ----------
    follicles, list
        A list of the measured follicle diameters.

    cutoff, {float, int}
        The minimum follicle diameter (in mm) considered to be viable for
        successful egg collection.
    """
    follicles = np.array(follicles)
    fols_above_cut_off = follicles[follicles > cutoff]
    mean = np.mean(fols_above_cut_off)
    total_count = len(follicles)
    above_count = len(fols_above_cut_off) # num above cutoff size
    percent  =  above_count / total_count        
    return [mean, above_count, percent]


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