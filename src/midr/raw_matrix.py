#!/usr/bin/python3

"""Compute the Irreproducible Discovery Rate (IDR) from NarrowPeaks files

This section of the project provides facilitites to handle NarrowPeaks files
and compute IDR on the choosen value in the NarrowPeaks columns
"""

from os import path, access, R_OK
from pathlib import PurePath
import pandas as pd
import numpy as np
import midr.log as log
from typing import Callable
from midr.auxiliary import benjamini_hochberg


def readfile(file_name: str):
    """
    read csv containing a matrix of row matching peaks score
    :param file_name:
    :return:
    """
    log.logging.info("%s", "reading matrix files")
    matrix_path = PurePath(file_name)
    assert path.isfile(str(matrix_path)), \
        "File {str(matrix_path)} doesn't exist"
    assert access(str(matrix_path), R_OK), \
        "File {str(matrix_path)} isn't readable"
    log.logging.info("%s", "reading " + str(matrix_path))
    return pd.read_csv(
        matrix_path,
        sep='\t',
        header=None,
        low_memory=True
    ).to_numpy()


def writefile(file_name: str, lidr: np.array, outdir: str):
    """
    Write output of IDR computation
    :param file_name: list of files names (str)
    :param lidr: np.array with local IDR score (columns correspond to bed files)
    :param outdir: output directory
    :return: nothing
    """
    log.logging.info("%s", "writing results")
    pd.DataFrame(
        {
            "lidr": lidr.tolist(),
            "idr": benjamini_hochberg(p_vals=lidr).tolist()
        }
    ).to_csv(
        PurePath(outdir).joinpath(
            "idr_" + PurePath(str(file_name)).name
        )
        , sep='\t',
        encoding='utf-8',
        header=False,
        index=False
    )


def process_matrix(file_name: str,
                outdir: str,
                idr_func: Callable[[np.array], np.array],
                threshold: float = 0.0001
                ):
    """
    Process a list of bed files names with the first names the merged bed files
    :param threshold:
    :param file_name: name of the matrix file
    :param outdir: output directory
    :param idr_func: idr function to apply
    :return: nothing
    """
    writefile(
        file_name=file_name,
        lidr=idr_func(
            readfile(
                file_name=file_name
            ),
            threshold=threshold
        ),
        outdir=outdir
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
