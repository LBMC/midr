#!/bin/python3

"""
script to compare midr to boley idr results
"""

from pathlib import PurePath
from typing import Callable

import pandas as pd
import numpy as np
import narrowpeak
import idr


def parse_boley(file_name: str,
                file_cols: list = None):
    """
    convert boley idr output file into midr input
    :param file_name: boley idr output file path
    :param file_cols: name of the narrowpeak columns
    :return:

    >>> # parse_boley(file_name="~/projects/gandrillon/midr/data/boleyidr2")
    """
    boley_bed = pd.read_csv(
        PurePath(file_name),
        sep='\t',
        header=None,
    )
    bed_number = int((boley_bed.shape[1] - (len(file_cols) + 2)) / 4)
    scores = list()
    for bed in range(bed_number):
        score_col = len(file_cols) + 2 + bed * 4 + 2
        scores.append(np.array(boley_bed[score_col].to_numpy()))
    scores = np.stack(scores, axis=-1)
    return scores, boley_bed


def process_bed(
        file_name: str,
        outdir: str,
        idr_func: Callable[[np.array], np.array] = idr.pseudo_likelihood,
        file_cols: list = None):
    """
    Process a list of bed files names with the first names the merged bed files
    :param file_name: file path
    :param outdir: output directory
    :param idr_func: idr function to apply
    :param file_cols: list of bed file columns
    :return: nothing

    >>> process_bed(
    ...     file_name="~/projects/gandrillon/midr/data/boleyidr2",
    ...     outdir="~/projects/gandrillon/midr/results/",
    ...     file_cols=narrowpeak.narrowpeaks_cols()
    ... )
    """
    scores, boley_bed = parse_boley(
        file_name=file_name,
        file_cols=file_cols
    )
    theta, lidr = idr_func(
        **{'x_score': scores,
           'log_name': "boley_comparison_t_0.0001_0.99_inv_process"}
    )
    print(theta)
    boley_bed.iloc[:, 11] = lidr
    output_name = PurePath(outdir).joinpath(
        "idr_" + PurePath(str(file_name)).name
    )
    boley_bed.assign(idr=lidr).to_csv(
        output_name, sep='\t',
        encoding='utf-8',
        header=False,
        index=False
    )


if __name__ == "__main__":
    import doctest
    doctest.testmod()
