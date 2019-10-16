#!/usr/bin/python3

"""Compute the Irreproducible Discovery Rate (IDR) from NarrowPeaks files

Implementation of the IDR methods for two or more replicates.

LI, Qunhua, BROWN, James B., HUANG, Haiyan, et al. Measuring reproducibility
of high-throughput experiments. The annals of applied statistics, 2011,
vol. 5, no 3, p. 1752-1779.

Given a list of peak calls in NarrowPeaks format and the corresponding peak
call for the merged replicate. This tool computes and appends a IDR column to
NarrowPeaks files.
"""

import math
from copy import deepcopy
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynverse import inversefunc
import log


def cov_matrix(m_sample, theta):
    """
    compute multivariate_normal covariance matrix

    >>> cov_matrix(3, {'rho':0.5, 'sigma':1})
    array([[1. , 0.5, 0.5],
           [0.5, 1. , 0.5],
           [0.5, 0.5, 1. ]])
    >>> cov_matrix(4, {'rho':0.5, 'sigma':2})
    array([[2., 1., 1., 1.],
           [1., 2., 1., 1.],
           [1., 1., 2., 1.],
           [1., 1., 1., 2.]])
    """
    cov = np.full(shape=(int(m_sample), int(m_sample)),
                  fill_value=float(theta['rho']) * float(theta['sigma']))
    np.fill_diagonal(a=cov,
                     val=float(theta['sigma']))
    return cov


def sim_multivariate_gaussian(n_value, m_sample, theta):
    """
    draw from a multivariate Gaussian distribution

    >>> sim_multivariate_gaussian(10, 2, \
        {'mu': 1, 'rho': 0.5, 'sigma': 1}).shape
    (10, 2)
    >>> np.mean(sim_multivariate_gaussian(10000, 1, \
         {'mu': 1, 'rho': 0.5, 'sigma': 1})[:,0]) > 0.9
    True
    >>> np.mean(sim_multivariate_gaussian(10000, 1, \
         {'mu': 1, 'rho': 0.5, 'sigma': 1})[:,0]) < 1.1
    True
    >>> np.var(sim_multivariate_gaussian(10000, 1, \
        {'mu': 1, 'rho': 0.5, 'sigma': 1})[:,0]) > 0.9
    True
    >>> np.var(sim_multivariate_gaussian(10000, 1, \
        {'mu': 1, 'rho': 0.5, 'sigma': 1})[:,0]) < 1.1
    True
    """
    cov = cov_matrix(m_sample=m_sample,
                     theta=theta)
    return np.random.multivariate_normal(mean=[float(theta['mu'])] *
                                         int(m_sample),
                                         cov=cov,
                                         size=int(n_value))


def sim_m_samples(n_value, m_sample, theta_0, theta_1):
    """
    simulate sample where position score are drawn from two different
    multivariate Gaussian distribution

    >>> sim_m_samples(100, 4, THETA_INIT, THETA_INIT)['X'].shape
    (100, 4)
    >>> len(sim_m_samples(100, 4, THETA_INIT, THETA_INIT)['K'])
    100
    """
    scores = sim_multivariate_gaussian(n_value=n_value,
                                       m_sample=m_sample,
                                       theta=theta_1)
    spurious = sim_multivariate_gaussian(n_value=n_value,
                                         m_sample=m_sample,
                                         theta=theta_0)
    k_state = list()
    for i in range(int(n_value)):
        k_state.append(True)
        if not bool(bernoulli.rvs(p=theta_1['pi'], size=1)):
            scores[i] = spurious[i]
            k_state[i] = False
    return {'X': scores, 'K': k_state}


def compute_rank(x_score):
    """
    transform x a n*m matrix of score into an n*m matrix of rank ordered by
    row.

    >>> compute_rank(np.array([[0,0],[10,30],[20,20],[30,10]]))
    array([[1, 1],
           [2, 4],
           [3, 3],
           [4, 2]])
    """
    plt.scatter(x_score[:, 0],
                x_score[:, 1])
    plt.ylabel('rank file 2')
    plt.xlabel('rank file 1')
    plt.savefig("test_plot_0.pdf")
    rank = np.empty_like(x_score)
    for j in range(x_score.shape[1]):
        # we want the rank to start at 1
        rank[:, j] = rankdata(x_score[:, j])

    plt.scatter(rank[:, 0],
                rank[:, 1])
    plt.ylabel('rank file 2')
    plt.xlabel('rank file 1')
    plt.savefig("test_plot_1.pdf")
    return rank


def compute_empirical_marginal_cdf(rank):
    """
    normalize ranks to compute empirical marginal cdf and scale by n / (n+1)

    >>> r = compute_rank(np.array([[0.0,0.0],[10.0,30.0],\
        [20.0,20.0],[30.0,10.0]]))
    >>> compute_empirical_marginal_cdf(r)
    array([[0.2, 0.2],
           [0.4, 0.8],
           [0.6, 0.6],
           [0.8, 0.4]])
    """
    x_score = np.empty_like(rank)
    n_value = float(rank.shape[0])
    m_sample = float(rank.shape[1])
    scaling_factor = n_value / (n_value + 1.0)
    for i in range(int(n_value)):
        for j in range(int(m_sample)):
            x_score[i][j] = (float(rank[i][j]) / n_value) * scaling_factor
    return x_score


def g_function(z_values, theta):
    """
    compute scalded Gaussian cdf for Copula
    """
    sigma = np.sqrt(float(theta['sigma']))
    #  z_norm = (float(z_values) - float(theta['mu'])) / sigma
    f_pi = float(theta['pi'])  # / sigma
    return f_pi * norm.cdf(float(z_values), loc=theta['mu'], scale=sigma) + \
        (1.0 - f_pi) * norm.cdf(float(z_values), loc=0, scale=1)


def compute_z_from_u(u_values, theta):
    """
    compute u_ij from z_ij via the G_j function

    >>> r = compute_rank(np.array([[0.0,0.0],[10.0,30.0],\
        [20.0,20.0],[30.0,10.0]]))
    >>> u = compute_empirical_marginal_cdf(r)
    >>> compute_z_from_u(u, {'mu': 1, 'rho': 0.5, 'sigma': 1, 'pi': 0.5})
    array([[-0.44976896, -0.44976896],
           [ 0.21303303,  1.44976896],
           [ 0.78696698,  0.78696698],
           [ 1.44976896,  0.21303303]])
    """
    #  fixed g function for given theta
    g_func = lambda x: g_function(x,
                                  theta=theta)
    #  compute inverse function of g_func
    g_m1 = lambda r: inversefunc(g_func,
                                 y_values=r,
                                 image=[0, 1],
                                 open_domain=False,
                                 domain=[min([-4, theta['mu'] - 4]),
                                         max([4, theta['mu'] + 4])],
                                 accuracy=0)
    z_values = np.empty_like(u_values)
    for i in range(u_values.shape[0]):
        for j in range(u_values.shape[1]):
            z_values[i][j] = g_m1(u_values[i][j])
    return z_values


def h_function(z_values, m_sample, theta):
    """
    compute the pdf of h0 or h1
    """
    try:
        cov = cov_matrix(m_sample=int(m_sample), theta=theta)
        x_values = multivariate_normal.pdf(x=z_values,
                                           mean=[float(theta['mu'])] *
                                           int(m_sample),
                                           cov=cov)
    except ValueError as err:
        log.LOGGER.exception("%s", "error: h_function: " + str(err))
        log.LOGGER.exception("%s", str(cov))
        log.LOGGER.exception("%s", str(theta))
    return pd.Series(x_values)


def e_step_k(z_values, theta):
    """
    compute expectation of Ki
    """
    h0_x = h_function(z_values=z_values,
                      m_sample=z_values.shape[1],
                      theta={'mu': 0,
                             'sigma': 1,
                             'rho': 0}
                      )
    h0_x = (1.0 - float(theta['pi'])) * h0_x
    h1_x = h_function(z_values=z_values,
                      m_sample=z_values.shape[1],
                      theta=theta
                      )
    h1_x = float(theta['pi']) * h1_x
    k_state = h1_x / (h1_x + h0_x)
    return k_state.to_list()


def local_idr(z_values, lidr, theta):
    """
    compute local IDR
    """
    h0_x = h_function(z_values=z_values,
                      m_sample=z_values.shape[1],
                      theta={'mu': 0,
                             'sigma': 1,
                             'rho': 0}
                      )
    h0_x = (1.0 - float(theta['pi'])) * h0_x
    h1_x = h_function(z_values=z_values,
                      m_sample=z_values.shape[1],
                      theta=theta
                      )
    h1_x = float(theta['pi']) * h1_x
    lidr = h0_x / (h1_x + h0_x)
    return lidr.to_list()


def m_step_pi(k_state):
    """
    compute maximization of pi
    """
    return float(sum(k_state)) / float(len(k_state))


def m_step_mu(z_values, k_state):
    """
    compute maximization of mu
    0 < mu
    """
    denominator = float(z_values.shape[1]) * float(sum(k_state))
    numerator = 0.0
    for i in range(z_values.shape[0]):
        for j in range(z_values.shape[1]):
            numerator += float(k_state[i]) * float(z_values[i][j])
    return numerator / denominator


def m_step_sigma(z_values, k_state, theta):
    """
    compute maximization of sigma
    """
    z_norm_sq = 0.0
    for i in range(z_values.shape[0]):
        for j in range(z_values.shape[1]):
            z_norm_sq += float(k_state[i]) * (float(z_values[i][j]) -
                                              float(theta['mu']))**2
    return (1.0 / (float(z_values.shape[1]) * float(sum(k_state)))) * z_norm_sq


def m_step_rho(z_values, k_state, theta):
    """
    compute maximization of rho
    0 < rho <= 1
    """
    nb_non_diag = float(z_values.shape[1])**2 - float(z_values.shape[1])
    z_norm_time = 0.0
    z_norm_time_i = 0.0
    for i in range(z_values.shape[0]):
        z_norm_time_i = 0.0
        for j in range(z_values.shape[1]):
            for k in range(z_values.shape[1]):
                if not k == j:
                    z_norm_time_i += (float(z_values[i][j]) -
                                      float(theta['mu'])) *\
                        (float(z_values[i][k]) - float(theta['mu']))
        z_norm_time += float(k_state[i]) * z_norm_time_i
    return z_norm_time / (nb_non_diag * theta['sigma'] * float(sum(k_state)))


def loglikelihood(z_values, k_state, theta):
    """
    Compute logLikelihood of the pseudo-data
    """
    try:
        h0_x = h_function(z_values=z_values,
                          m_sample=z_values.shape[1],
                          theta={'mu': 0,
                                 'sigma': 1,
                                 'rho': 0}
                          )
        h1_x = h_function(z_values=z_values,
                          m_sample=z_values.shape[1],
                          theta=theta
                          )
        logl = 0.0
        for i in range(z_values.shape[0]):
            logl += (1.0-float(k_state[i])) * (math.log(1-theta['pi']) +
                                               math.log(h0_x[i]))
            logl += float(k_state[i]) * (math.log(theta['pi']) +
                                         math.log(h1_x[i]))
        return logl
    except ValueError as err:
        log.LOGGER.exception("%s", "error: logLikelihood: " + str(err))
        log.LOGGER.exception("%s", str(h1_x[i]))
        log.LOGGER.exception("%s", str(theta))
        quit(-1)


def delta(theta_t0, theta_t1, threshold, logl):
    """
    compute the maximal variation between t0 and t1 for the estimated
    parameters
    """
    if logl == -np.inf:
        return True
    for parameters in theta_t0:
        if abs(theta_t0[parameters] - theta_t1[parameters]) > threshold:
            return True
    return False


def em_pseudo_data(z_values,
                   logger,
                   theta,
                   k_state,
                   threshold=0.001):
    """
    EM optimization of theta for pseudo-data
    """
    theta_t0 = deepcopy(theta)
    theta_t1 = deepcopy(theta)
    logl_t0 = 0.0
    logl_t1 = -np.inf
    while delta(theta_t0, theta_t1, threshold, logl_t1):
        logl_t0 = logl_t1
        del theta_t0
        theta_t0 = deepcopy(theta_t1)
        k_state = e_step_k(z_values=z_values,
                           theta=theta_t1)
        theta_t1['pi'] = m_step_pi(k_state=k_state)
        theta_t1['mu'] = m_step_mu(z_values=z_values,
                                   k_state=k_state)
        theta_t1['sigma'] = m_step_sigma(z_values=z_values,
                                         k_state=k_state,
                                         theta=theta_t1)
        theta_t1['rho'] = m_step_rho(z_values=z_values,
                                     k_state=k_state,
                                     theta=theta_t1)
        logl_t1 = loglikelihood(z_values=z_values,
                                k_state=k_state,
                                theta=theta_t1)
        if logl_t1 - logl_t0 < 0.0:
            log.LOGGER.debug("%s",
                                    "warning: EM decreassing logLikelihood \
                                    rho: " +
                                    str(logl_t1 - logl_t0))
            log.LOGGER.debug("%s", str(theta_t1))
            return (theta_t0, k_state, logger)
        logger = log.add_log(log=logger,
                                 theta=theta_t1,
                                 logl=logl_t1,
                                 pseudo=False)
    return (theta_t1, k_state, logger)


def pseudo_likelihood(x_score, threshold=0.001, log_name=""):
    """
    pseudo likelhood optimization for the copula model parameters
    """
    theta_t0 = deepcopy(THETA_INIT)
    theta_t1 = deepcopy(THETA_INIT)
    k_state = [0.0] * int(x_score.shape[0])
    u_values = [0.0] * int(x_score.shape[0])
    z_values = [0.0] * int(x_score.shape[0])
    lidr = [0.0] * int(x_score.shape[0])
    logger = {'logl': list(),
           'pi': list(),
           'mu': list(),
           'sigma': list(),
           'rho': list(),
           'pseudo_data': list()}
    logl_t1 = -np.inf
    u_values = compute_empirical_marginal_cdf(compute_rank(x_score))
    while delta(theta_t0, theta_t1, threshold, logl_t1):
        del theta_t0
        theta_t0 = deepcopy(theta_t1)
        z_values = compute_z_from_u(u_values=u_values,
                                    theta=theta_t1)
        (theta_t1, k_state, logger) = em_pseudo_data(z_values=z_values,
                                                  log=logger,
                                                  k_state=k_state,
                                                  theta=theta_t1,
                                                  threshold=threshold)
        lidr = local_idr(z_values=z_values,
                         lidr=lidr,
                         theta=theta_t1)
        logl_t1 = loglikelihood(z_values=z_values,
                                k_state=k_state,
                                theta=theta_t1)
        logger = log.add_log(log=logger,
                                 theta=theta_t1,
                                 logl=logl_t1,
                                 pseudo=True)
        log.plot_log(log, str(log_name) + "_log.pdf")
        log.plot_classif(x_score,
                                u_values,
                                z_values,
                                lidr,
                                str(log_name) + "_classif.pdf")
        log.LOGGER.debug("%s", str(theta_t1))
    return (theta_t1, lidr)

#  THETA_TEST_0 = {'pi': 0.6, 'mu': 0.0, 'sigma': 1.0, 'rho': 0.0}
#  THETA_TEST_1 = {'pi': 0.6, 'mu': 4.0, 'sigma': 3.0, 'rho': 0.75}
#  THETA_TEST = {'pi': 0.2,
#                'mu': THETA_TEST_1['mu'] - THETA_TEST_0['mu'],
#                'sigma': THETA_TEST_0['sigma'] / THETA_TEST_1['sigma'],
#                'rho': 0.75}
#
#  DATA = sim_m_samples(n_value=1000,
#                       m_sample=2,
#                       theta_0=THETA_TEST_0,
#                       theta_1=THETA_TEST_1)
#  (THETA_RES, LIDR) = pseudo_likelihood(DATA["X"],
#                                           threshold=0.01,
#                                           log_name=str(THETA_TEST))
#  print(THETA_TEST)
#
#  plt.subplot(1, 1, 1)
#  plt.scatter(DATA['K'], K, c=LIDR)
#  plt.ylabel('k')
#  plt.savefig("k_vs_estK_" + str(THETA_TEST) + ".pdf")

#  test = NarrowPeaks(file_merge="data/test/c1_merge.narrowPeak",
#                     file_names=["data/test/c1_r1.narrowPeak",
#                                 "data/test/c1_r2.narrowPeak"])



THETA_INIT = {'pi': 0.5,
              'mu': 0.0,
              'sigma': 1.0,
              'rho': 0.9}
