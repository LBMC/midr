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

def eulerian(np.int n, np.int m):
    """
    compute eulerian numbers
    :param n:
    :param m:
    :return:
    """
    cdef int i
    cdef int j
    dp = np.full((n + 1, m + 1), 0)
    # For each row from 1 to n
    for i in range(1, n + 1):
        # For each column from 0 to m
        for j in range(0, m + 1):
            # If i is greater than j
            if i > j:
                # If j is 0, then make that
                # state as 1.
                if j == 0:
                    dp[i, j] = 1
                # basic recurrence relation.
                else:
                    dp[i, j] = (((i - j) * dp[i - 1, j - 1]) +
                                ((j + 1) * dp[i - 1, j]))
    return dp[n, m]


def eulerian_all(np.int n):
    """
    compute eulerian number
    :param n:
    :return:
    >>> eulerian_all(10)
    array([1.000000e+00, 1.013000e+03, 4.784000e+04, 4.551920e+05,
           1.310354e+06, 1.310354e+06, 4.551920e+05, 4.784000e+04,
           1.013000e+03, 1.000000e+00])
    """
    cdef int i
    y = np.zeros(shape=n)
    cdef np.float64_t[::] res = y
    for i in range(n):
        res[i] = eulerian(n, i)
    return res


def log1mexpvec(np.float64_t[::] x):
    """
    compute log(1-exp(-a)
    :param x:
    :return:
    """
    cdef int i
    y = np.empty_like(x)
    cdef np.float64_t[::] res = y
    for i in range(len(x)):
        eps = np.log(2.0)
        if x[i] <= eps:
            res[i] = np.log(-np.expm1(-x[i]))
        else:
            res[i] = np.log1p(-np.exp(-x[i]))
    return res


def polyneval(np.float64_t[::] coef, np.float64_t[::] x, negative = False):
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
            if negative:
                res[i] = -res[i]
    return res


def minus_vec(np.float64_t[::] x):
    """
    return -x
    """
    cdef int i
    y = np.zeros([len(x)], dtype=np.float64)
    cdef np.float64_t[::] res = y
    for i in range(len(x)):
        res[i] = -x[i]
    return res


def polylog(np.float64_t[::] z, np.float64_t s, is_log_z=False):
    """
    :param z:
    :param s:
    :param is_log_z:
    :return:
    >>> polylog(np.array([0.01556112, 0.00108968, 0.00889932]), -2)
    array([-4.1004881 , -6.81751129, -4.68610299])
    """
    n = -int(s)
    eun = eulerian_all(n)
    if is_log_z:
        w = z
        z = np.exp(w)
        return np.array(np.log(polyneval(eun, z)) + w - (n + 1.0) * log1mexpvec(
            w, negative=True), dtype=np.float64)
    else:
        return np.array(np.log(polyneval(eun, z)) + np.log(z) - (n + 1.0) *
                               np.log1p(minus_vec(z)), dtype=np.float64)
