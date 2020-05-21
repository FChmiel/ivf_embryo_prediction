"""
A set of utlity functions used throughout the ivf_embryo_prediction.

ivf_embryo_prediction, Machine-learnt models for predicting chance
of suitable embryo for D5 transfer or freezing.

Copyright (C) 2020  F. P. Chmiel

Email: F.P.Chmiel@soton.ac.uk

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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