#!/usr/bin/python3

"""Compute the Irreproducible Discovery Rate (IDR) from NarrowPeaks files

This section of the project provides facilitites to handle NarrowPeaks files
and compute IDR on the choosen value in the NarrowPeaks columns
"""

from scipy.stats import rankdata


def benjamini_hochberg(p_vals):
    """
    compute fdr from pvalues
    :param p_vals:
    :return:
    """
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1
    return fdr


