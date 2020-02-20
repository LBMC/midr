"""Compute the Irreducible Discovery Rate (IDR) from NarrowPeaks files

Implementation of the IDR methods for two or more replicates.

LI, Qunhua, BROWN, James B., HUANG, Haiyan, et al. Measuring reproducibility
of high-throughput experiments. The annals of applied statistics, 2011,
vol. 5, no 3, p. 1752-1779.

Given a list of peak calls in NarrowPeaks format and the corresponding peak
call for the merged replicate. This tool computes and appends a IDR column to
NarrowPeaks files.
"""

cimport cython
cimport numpy as np
import numpy as np

def polyneval(np.float64_t[::] coef, np.float64_t[::] x):
    """
    :param coef:
    :param x:
    :return:
    >>> polyneval(eulerian_all(10), np.array([-4, -3]))
    array([1.12058925e+08, 9.69548800e+06])
    """
    cdef int i
    cdef int j
    y = np.zeros([len(x)], dtype=np.float64)
    cdef np.float64_t[::] res = y
    for i in range(len(x)):
        for j in range(len(coef)):
            res[i] = res[i] * x[i] + coef[j]
    return res
