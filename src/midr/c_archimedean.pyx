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
    Return euleriannumber A(n, m)
    :param n:
    :param m:
    :return:
    """
    if (m >= n or n == 0):
        return 0;

    if (m == 0):
        return 1;

    return ((n - m) * eulerian(n - 1, m - 1) +
            (m + 1) * eulerian(n - 1, m))


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
    cdef int n = len(x)
    cdef float eps = np.log(2.0)
    cdef np.float64_t[::] res = np.empty_like(x)
    for i in range(n):
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
    cdef int n = len(x)
    cdef int m = len(coef)
    cdef float r
    cdef float xi
    cdef np.float64_t[::] res = np.zeros([len(x)], dtype=np.float64)
    for i in range(n):
        r = x[i]
        xi = x[i]
        if m == 0:
            r = 0.
        else:
            j = m-1
            r = coef[j]
            for j in range(m-1, 0, -1):
                r = r * xi + coef[j]
        if negative:
            res[i] = -r
        else:
            res[i] = r
    return res


def minus_vec(np.float64_t[::] x):
    """
    return -x
    """
    cdef int i
    cdef np.float64_t[::] res = np.zeros([len(x)], dtype=np.float64)
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
    if is_log_z:
        w = z
        z = np.exp(w)
        return np.array(
            np.log(polyneval(eulerian_all(n), z)) +
            w - (n + 1.0) * log1mexpvec(minus_vec(w))
            ,
            dtype=np.float128)
    else:
        return np.array(
            np.log(polyneval(eulerian_all(n), z)) +
            np.log(z) - (n + 1.0) * np.log1p(minus_vec(z)),
            dtype=np.float128)
