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
import multiprocessing as mp
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
from scipy.stats import poisson
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.special import factorial
from scipy.special import binom
import numpy as np
from mpmath import polylog
import pandas as pd
import midr.log as log

import matplotlib.pyplot as plt


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


def sim_m_samples(n_value, m_sample,
                  theta_0={'pi': 0.2, 'mu': 0.0, 'sigma': 1.0, 'rho': 0.0},
                  theta_1={'pi': 0.6, 'mu': 0.0, 'sigma': 1.0, 'rho': 0.0}):
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
    rank = np.empty_like(x_score)
    for j in range(x_score.shape[1]):
        # we want the rank to start at 1
        rank[:, j] = rankdata(x_score[:, j], method="ordinal")
    return rank


def compute_empirical_marginal_cdf(rank):
    """
    normalize ranks to compute empirical marginal cdf and scale by n / (n+1)

    >>> r = compute_rank(np.array(
    ...    [[0.0,0.0],
    ...    [10.0,30.0],
    ...    [20.0,20.0],
    ...    [30.0,10.0]]))
    >>> compute_empirical_marginal_cdf(r)
    array([[0.99  , 0.99  ],
           [0.7425, 0.2475],
           [0.495 , 0.495 ],
           [0.2475, 0.7425]])
    """
    x_score = np.empty_like(rank)
    n_value = float(rank.shape[0])
    m_sample = float(rank.shape[1])
    # scaling_factor = n_value / (n_value + 1.0)
    # we want a max value of 0.99
    scaling_factor = n_value / 0.99 - n_value
    scaling_factor = n_value / (n_value + scaling_factor)
    for i in range(int(n_value)):
        for j in range(int(m_sample)):
            x_score[i][j] = (1.0 - (float(rank[i][j] - 1) / n_value)) * \
                            scaling_factor
    return x_score


def g_function(z_values, theta):
    """
    compute scalded Gaussian cdf for Copula
    """
    sigma = np.sqrt(float(theta['sigma']))
    f_pi = float(theta['pi'])  # / sigma
    return f_pi * norm.cdf(float(z_values),
                           loc=float(theta['mu']),
                           scale=np.sqrt(float(theta['sigma']))) + \
           (1.0 - f_pi) * norm.cdf(float(z_values), loc=0.0, scale=1.0)


def compute_grid(theta,
                 function=g_function,
                 size=1000,
                 z_start=-4.0,
                 z_stop=4.0):
    """
    compute a grid of function(z_values) from z_start to z_stop
    :param function: function
    :param theta: function parameters
    :param size: size of the grid
    :param z_start: start of the z_values
    :param z_stop: stop of the z_values
    :return: pd.array of 'z_values' paired with 'u_values'

    >>> compute_grid(
    ...    theta={'pi': 0.6, 'mu': 1.0, 'sigma': 2.0, 'rho': 0.0},
    ...    size=4
    ... )
       z_values  u_values
    0 -4.000000  0.000135
    1 -1.333333  0.066173
    2  1.333333  0.719416
    3  4.000000  0.989819
    """
    z_grid = np.linspace(
        start=z_start,
        stop=z_stop,
        num=size
    )
    u_grid = [0.0] * len(z_grid)
    for i in range(len(z_grid)):
        u_grid[i] = function(z_values=z_grid[i], theta=theta)
    return pd.DataFrame({'z_values': z_grid.tolist(), 'u_values': u_grid})


def z_from_u_worker(q: mp.JoinableQueue, function, grid, u_values, z_values):
    """
    z_from_u unit function in case of multiprocessing
    :param q:
    :param function:
    :param grid:
    :param u_values:
    :param z_values:
    :return:
    """
    while not q.empty():
        i = q.get()
        a_loc = grid.loc[grid['u_values'] <= u_values[i]]
        a_loc = a_loc.iloc[len(a_loc) - 1:len(a_loc)].index[0]
        b_loc = grid.loc[grid['u_values'] >= u_values[i]].index[0]
        z_values[i] = brentq(
            f=lambda x: function(x, u_values[i]),
            a=grid.iloc[a_loc, 0],
            b=grid.iloc[b_loc, 0]
        )
        q.task_done()


def z_from_u(u_values, function, grid, thread_num=mp.cpu_count()):
    """
    Compute z_values from u_values
    :param u_values: list of u_values
    :param function: g_function
    :param theta: g_function parameter
    :return: list of z_value

    >>> z_from_u(
    ...    u_values=[0.2, 0.3, 0.5, 0.9],
    ...    function=lambda x, y: y - g_function(
    ...            z_values=x,
    ...            theta={'pi': 0.6, 'mu': 1.0, 'sigma': 2.0, 'rho': 0.0}
    ...        ),
    ...    grid=compute_grid(
    ...        theta={'pi': 0.6, 'mu': 1.0, 'sigma': 2.0, 'rho': 0.0},
    ...        size=20
    ...    )
    ... )
    [-0.5429962873458862, -0.1535404920578003, 0.5210787653923035, \
2.3994555473327637]
    """
    z_values = [0.0] * len(u_values)
    if thread_num == 0:
        for i in range(len(u_values)):
            a_loc = grid.loc[grid['u_values'] <= u_values[i]]
            a_loc = a_loc.iloc[len(a_loc) - 1:len(a_loc)].index[0]
            b_loc = grid.loc[grid['u_values'] >= u_values[i]].index[0]
            z_values[i] = brentq(
                f=lambda x: function(x, u_values[i]),
                a=grid.iloc[a_loc, 0],
                b=grid.iloc[b_loc, 0]
            )
    else:
        q = mp.JoinableQueue()
        shared_z_values = mp.Array('f', [0.0] * len(u_values), lock=False)
        list(map(lambda x: q.put(x), range(len(u_values))))
        worker = map(
            lambda x: mp.Process(
                target=z_from_u_worker,
                args=(q, function, grid, u_values, shared_z_values),
                name="z_from_u_" + str(x),
                daemon=True
            ),
            range(thread_num)
        )
        list(map(lambda x: x.start(), worker))
        q.join()
        list(map(lambda x: x.join(), worker))
        z_values = list(shared_z_values)
    return z_values


def compute_z_from_u(u_values, theta):
    """
    compute u_ij from z_ij via the G_j function

    >>> r = compute_rank(np.array([[0.0,0.0],[10.0,30.0],\
        [20.0,20.0],[30.0,10.0]]))
    >>> u = compute_empirical_marginal_cdf(r)
    >>> compute_z_from_u(u, {'mu': 1, 'rho': 0.5, 'sigma': 1, 'pi': 0.5})
    array([[ 3.07591891,  3.07591891],
           [ 1.23590529, -0.27110639],
           [ 0.48579773,  0.48579773],
           [-0.27110639,  1.23590529]])
    """
    grid = compute_grid(
        theta=theta,
        function=g_function,
        size=1000,
        z_start=norm.ppf(np.amin(u_values), loc=-abs(theta['mu'])) - 1.0,
        z_stop=norm.ppf(np.amax(u_values), loc=abs(theta['mu'])) + 1.0
    )
    z_values = np.empty_like(u_values)
    for j in range(u_values.shape[1]):
        z_values[:, j] = z_from_u(
            u_values=u_values[:, j],
            function=lambda x, y: y - g_function(
                z_values=x,
                theta=theta
            ),
            grid=grid)
    return z_values


def h_function(z_values, m_sample, theta):
    """
    compute the pdf of h0 or h1
    """
    cov = cov_matrix(m_sample=int(m_sample), theta=theta)
    try:
        x_values = multivariate_normal.pdf(x=z_values,
                                           mean=[float(theta['mu'])] *
                                                int(m_sample),
                                           cov=cov)
        return pd.Series(x_values)
    except ValueError as err:
        log.LOGGER.exception("%s", "error: h_function: " + str(err))
        log.LOGGER.exception("%s", str(cov))
        log.LOGGER.exception("%s", str(theta))


def e_step_k(z_values, theta):
    """
    compute expectation of Ki
    """
    h0_x = h_function(z_values=z_values,
                      m_sample=z_values.shape[1],
                      theta={'mu': 0.0,
                             'sigma': 1.0,
                             'rho': 0.0}
                      )
    h0_x *= 1.0 - float(theta['pi'])
    h1_x = h_function(z_values=z_values,
                      m_sample=z_values.shape[1],
                      theta=theta
                      )
    h1_x *= float(theta['pi'])
    k_state = h1_x / (h1_x + h0_x)
    return k_state.to_list()


def local_idr(z_values, theta):
    """
    compute local IDR
    """
    h0_x = h_function(z_values=z_values,
                      m_sample=z_values.shape[1],
                      theta={'mu': 0.0,
                             'sigma': 1.0,
                             'rho': 0.0}
                      )
    h0_x *= (1.0 - float(theta['pi']))
    h1_x = h_function(z_values=z_values,
                      m_sample=z_values.shape[1],
                      theta=theta
                      )
    h1_x *= float(theta['pi'])
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
                                              float(theta['mu'])) ** 2.0
    return (1.0 / (float(z_values.shape[1]) * float(sum(k_state)))) * z_norm_sq


def m_step_rho(z_values, k_state, theta):
    """
    compute maximization of rho
    0 < rho <= 1
    """
    nb_non_diag = float(z_values.shape[1]) ** 2.0 - float(z_values.shape[1])
    z_norm_time = 0.0
    for i in range(z_values.shape[0]):
        z_norm_time_i = 0.0
        for j in range(z_values.shape[1]):
            for k in range(z_values.shape[1]):
                if not k == j:
                    z_norm_time_i += (float(z_values[i][j]) -
                                      float(theta['mu'])) * \
                                     (float(z_values[i][k]) - float(
                                         theta['mu']))
        z_norm_time += float(k_state[i]) * z_norm_time_i
    return z_norm_time / (nb_non_diag * theta['sigma'] * float(sum(k_state)))


def loglikelihood(z_values, k_state, theta):
    """
    Compute logLikelihood of the pseudo-data
    """
    h1_x = [0.0]
    i = 0
    try:
        h0_x = h_function(z_values=z_values,
                          m_sample=z_values.shape[1],
                          theta={'mu': 0.0,
                                 'sigma': 1.0,
                                 'rho': 0.0}
                          )
        h1_x = h_function(z_values=z_values,
                          m_sample=z_values.shape[1],
                          theta=theta
                          )
        logl = 0.0
        for i in range(z_values.shape[0]):
            logl += (1.0 - float(k_state[i])) * (
                    math.log(1.0 - float(theta['pi'])) + math.log(h0_x[i]))
            logl += float(k_state[i]) * (
                    math.log(float(theta['pi'])) + math.log(h1_x[i]))
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
    >>> THETA_TEST = {'pi': 0.2,
    ...               'mu': 2.0,
    ...               'sigma': 3.0,
    ...               'rho': 0.65}
    >>> DATA = sim_m_samples(n_value=1000,
    ...                      m_sample=2,
    ...                      theta_1=THETA_TEST)
    >>> (THETA_RES, KSTATE, LIDR) = em_pseudo_data(
    ...    z_values=DATA["X"],
    ...    logger={
    ...        'logl': list(),
    ...        'pi': list(),
    ...        'mu': list(),
    ...        'sigma': list(),
    ...        'rho': list(),
    ...        'pseudo_data': list()
    ...    },
    ...    theta=THETA_TEST,
    ...    k_state=[0.0] * DATA['X'].shape[0],
    ...    threshold=0.01)
    >>> abs(THETA_RES['pi'] - THETA_TEST['pi']) < 0.2
    True
    >>> abs(THETA_RES['mu'] - THETA_TEST['mu']) < 0.2
    True
    >>> abs(THETA_RES['sigma'] - THETA_TEST['sigma']) < 1.0
    True
    >>> abs(THETA_RES['rho'] - THETA_TEST['rho']) < 0.2
    True
    """
    theta_t0 = deepcopy(theta)
    theta_t1 = deepcopy(theta)
    logl_t1 = -np.inf
    while delta(theta_t0, theta_t1, threshold, logl_t1):
        logl_t0 = logl_t1
        del theta_t0
        theta_t0 = deepcopy(theta_t1)
        k_state = e_step_k(
            z_values=z_values,
            theta=theta_t1
        )
        theta_t1['pi'] = m_step_pi(
            k_state=k_state
        )
        theta_t1['mu'] = m_step_mu(
            z_values=z_values,
            k_state=k_state
        )
        theta_t1['sigma'] = m_step_sigma(
            z_values=z_values,
            k_state=k_state,
            theta=theta_t1
        )
        theta_t1['rho'] = m_step_rho(
            z_values=z_values,
            k_state=k_state,
            theta=theta_t1
        )
        logl_t1 = loglikelihood(
            z_values=z_values,
            k_state=k_state,
            theta=theta_t1
        )
        if logl_t1 - logl_t0 < 0.0:
            log.LOGGER.debug("%s",
                             "warning: EM decreassing logLikelihood \
                             rho: " +
                             str(logl_t1 - logl_t0))
            log.LOGGER.debug("%s", str(theta_t1))
            return theta_t0, k_state, logger
        logger = log.add_log(
            log=logger,
            theta=theta_t1,
            logl=logl_t1,
            pseudo=False
        )
    return theta_t1, k_state, logger


def lsum(x):
    """
    compute log sum_i x_i
    :param x:
    :return:
    """
    lx = np.log(x)
    x_max = max(lx)
    return x_max + np.log(np.sum(np.exp(lx - x_max), axis=0))


def lssum(x_values):
    """
    compute log sum_i x_i with sign
    :param x_values:
    :return:
    """
    b_i = np.sort(np.log(abs(x_values)))
    b_max = max(b_i)
    results = 0.0
    for i in range(x_values.shape[0]):
        if b_i[i] >= 0.0:
            results += np.exp(b_i[i] - b_max)
        else:
            results -= np.exp(b_i[i] - b_max)
    return b_max + np.log(results)

def log1mexp(x):
    """
    compute log(1-exp(-a)
    :param x:
    :return:
    """
    def mapping_function(x):
        if x <= np.log(2.0):
            return np.log(-np.expm1(-x))
        else:
            return np.log1p(-np.exp(-x))
    mapping_function = np.vectorize(mapping_function)
    return mapping_function(x)

def log1pexp(x):
    """
    compute log(1 + exp(x))
    :param x:
    :return:
    """
    return np.logaddexp(0.0, x)


def log_copula_franck_cdf(u_values, theta):
    """
    compute log franck copula cdf
    :param u_values:
    :param theta:
    :return:
    >>> log_copula_franck_cdf(np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ]),
    ...    0.1)
    array([0.246109  , 0.33384317, 0.27751253, 0.252856  , 0.39741306,
           0.23508275, 0.3151825 , 0.38446148, 0.29659567, 0.29323393])
    """

    def h_f(ui_values, theta):
        result = (1.0 - np.exp(-theta)) ** (1.0 - ui_values.shape[0])
        for j in range(ui_values.shape[0]):
            result *= 1.0 - np.exp(-theta * ui_values[j])
        return result

    def lh_f(ui_values, theta):
        result = (1.0 - ui_values.shape[0]) * np.log(1.0 - np.exp(-theta))
        for j in range(ui_values.shape[0]):
            result += np.log(1.0 - np.exp(-theta * ui_values[j]))
        return result

    def lpolylog(ui_values, theta):
        return np.log(float(polylog(ui_values.shape[0], h_f(ui_values, theta))))

    lc_theta = np.empty_like(u_values[:, 0])
    for i in range(u_values.shape[0]):
        lc_theta[i] = (u_values.shape[0] - 1)
        lc_theta[i] *= np.log(theta) - np.log(1 - np.exp(-theta))
        lc_theta[i] += lpolylog(u_values[i, :], theta)
        lc_theta[i] -= theta * sum(u_values[i, :])
        lc_theta[i] -= lh_f(u_values[i, :], theta)
    return lc_theta


def copula_franck_cdf(u_values, theta):
    """
    compute franck copula cdf
    :param u_values:
    :param theta:
    :return:
    >>> copula_franck_cdf(np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ]),
    ...    0.1)
    array([-0.02974577, -0.0049061 , -0.0172447 , -0.01165619, -0.00016851,
           -0.03069314, -0.00659961, -0.00044277, -0.00958698, -0.00923634])
    """
    copula = np.empty_like(u_values[:, 0])
    for i in range(u_values.shape[0]):
        copula[i] = 1
        for j in range(u_values.shape[1]):
            copula[i] *= np.exp(-float(theta) * float(u_values[i, j])) - 1.0
        copula[i] /= np.exp(-float(theta)) - 1.0
        copula[i] = -1.0 / float(theta) * np.log(1.0 + copula[i])
    return copula



def log_copula_clayton_cdf(u_values, theta):
    """
    compute franck copula cdf
    :param u_values:
    :param theta:
    :return:
    >>> log_copula_clayton_cdf(np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ]),
    ...    0.1)
    array([-18.38984653, -23.19518979, -20.04813123, -19.32992577,
           -25.88637098, -18.12908708, -22.17537644, -25.48058402,
           -21.23116191, -21.17066147])
    """

    def sum_k(ui_values, theta):
        result = 0.0
        for j in range(ui_values.shape[0]):
            result += np.log(theta * float(j + 1) + 1.0)
        return result

    def t_theta(ui_values, theta):
        results = 0.0
        for j in range(ui_values.shape[0]):
            results += (1.0 + theta * ui_values[j]) ** -(1.0 / theta)
        return results * (float(u_values.shape[1]) + 1.0 / theta)

    lc_theta = np.empty_like(u_values[:, 0])
    for i in range(u_values.shape[0]):
        lc_theta[i] = sum_k(u_values[i, :], theta)
        lc_theta[i] -= (1 + theta) * sum(np.log(u_values[i, :]))
        lc_theta[i] -= t_theta(u_values[i, :], theta)
    return lc_theta


def log_pdf_diag_copula(u_values, theta, iPsi, absdPsi, absdiPsi):
    """
    compute pdf of the diagonal copula
    :param u_values:
    :param theta:
    :param iPsi:
    :param absdPsi:
    :param absdiPsi:
    :param is_log:
    :return:
    """
    y = diag_copula(u_values)
    for i in range(y.shape[0]):
        y[i] = np.log(u_values.shape[1]) + absdPsi(
        u_values.shape[1] * iPsi(y[i], theta), theta
        ) + absdiPsi(y[i], theta)
    return y

def max_ddelta(u_values, iPsi, absdPsi, absdiPsi, constraint, plot=False):
    """
    find theta using DMLE from diagonal pdf
    :param u_values:
    :param iPsi:
    :param absdPsi:
    :param absdiPsi:
    :param plot:
    :return:
    """
    def log_ddelta(theta, u_values):
        return -np.sum(log_pdf_diag_copula(
            u_values=u_values,
            theta=theta,
            absdPsi=absdPsi,
            iPsi=iPsi,
            absdiPsi=absdiPsi
        ))
    res = minimize(
        fun=lambda x: log_ddelta(x, u_values),
        x0=0.5,
        constraints=constraint
    )
    if res.success:
        return res.x[0]
    else:
        return np.NaN


def DMLE_copula_clayton(u_values):
    """
    compute clayton theta with DMLE
    :param u_values:
    :param theta:
    :return:
    >>> DMLE_copula_clayton( np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ])
    ... )
    0.2740632857071272
    """
    def absdPsi(x, theta):
        alpha = 1.0 / theta
        if theta > 0.0:
            return gammaln(alpha + 1.0) - gammaln(alpha) - (1.0 + alpha) * \
            np.log1p(x)
        return -(gammaln(1.0+alpha) - gammaln(alpha)) - (1.0 + alpha) * \
        np.log1p(-x)

    def iPsi(x, theta):
        return np.sign(theta) * (x ** (-theta) - 1.0)

    def absdiPsi(x, theta):
        return np.log(theta)-(1.0+theta)*np.log(x)

    return max_ddelta(
        u_values=u_values,
        absdPsi=absdPsi,
        iPsi=iPsi,
        absdiPsi=absdiPsi,
        constraint=[{'type': 'ineq', 'fun': lambda x: 1e-14},
                    {'type': 'ineq', 'fun': lambda x: 1000 - x}],
        plot=True
    )

def DMLE_copula_franck(u_values):
    """
    compute franck theta with DMLE
    :param u_values:
    :param theta:
    :return:
    >>> DMLE_copula_franck(np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ])
    ... )
    3.0737030577878235
    """
    def absdPsi(x, theta):
        w = log1mexp(theta) - x
        li = w - log1mexp(-w)
        return li - np.log(theta)

    def iPsi(x, theta):
        return -np.log1p((np.exp(-x * theta) - np.exp(-theta)) / np.expm1(
            -theta))

    def absdiPsi(x, theta):
        y = x * theta
        y = y + log1mexp(y)
        return np.log(theta) - y

    return max_ddelta(
        u_values=u_values,
        absdPsi=absdPsi,
        iPsi=iPsi,
        absdiPsi=absdiPsi,
        constraint=[{'type': 'ineq', 'fun': lambda x: x - 1e-14},
                    {'type': 'ineq', 'fun': lambda x: 745 - x}]
    )


def copula_clayton_cdf(u_values, theta):
    """
    compute franck copula cdf
    :param u_values:
    :param theta:
    :return:
    >>> copula_clayton_cdf(np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ]),
    ...    0.1)
    array([0.3271503 , 0.06358001, 0.19507278, 0.1269139 , 0.00426169,
           0.33473455, 0.08108552, 0.00869809, 0.11257681, 0.10827238])
    """
    copula = np.empty_like(u_values[:, 0])
    for i in range(u_values.shape[0]):
        copula[i] = 0
        for j in range(u_values.shape[1]):
            copula[i] += float(u_values[i, j]) ** (-float(theta))
        copula[i] += - float(u_values.shape[1]) + 1.0
        copula[i] **= -1.0 / float(theta)
    return copula

def ipsi_clayton(x, theta, is_log=False):
    """
    compute Clayton iPsi function
    :param x:
    :param theta:
    :return:
    >>> ipsi_clayton(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=True)
    array([[-1.68970664, -0.9046472 , -4.60419006],
           [-1.14586247, -0.25021206, -1.41715244],
           [-3.05365563, -2.88358502, -0.36029677],
           [ 0.05847131, -1.77732101, -2.33483902],
           [-1.58955921,  0.13757446, -2.34661855],
           [-2.47950927, -1.54631906, -1.96358583],
           [-3.88890049, -0.95569455, -2.52719433],
           [-1.25734211, -3.25226671, -2.91834924],
           [-1.05452015, -0.85128374, -2.14210478],
           [-2.47465594, -2.58334647, -1.40228078]])
    """
    if is_log:
        return np.log(np.sign(theta) * (x ** (-theta) - 1.0))
    return np.sign(theta) * (x ** (-theta) - 1.0)

def psi_clayton(x, theta):
    """
    compute Clayton Psi function
    :param x:
    :param theta:
    :return:
    >>> psi_clayton(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2)
    array([[0.16797341, 0.43186024, 0.03533839],
           [0.32574507, 0.7608775 , 0.23335089],
           [0.05379659, 0.058921  , 0.70980892],
           [0.87552668, 0.15183754, 0.0869198 ],
           [0.18914097, 0.89736349, 0.08605411],
           [0.07726802, 0.19926139, 0.12383307],
           [0.03999973, 0.40771118, 0.07451141],
           [0.28422656, 0.04910704, 0.05777555],
           [0.36343815, 0.45791558, 0.10349543],
           [0.07755952, 0.07150121, 0.23766697]])
    """
    return np.maximum(1.0 + np.sign(theta) * x, 0.0) ** (-1.0 / theta)

def pdf_clayton(u_values, theta, is_log=False):
    """
    compute clayton copula pdf
    :param u_values:
    :param theta:
    :return:
    >>> pdf_clayton(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=True)
    array([-0.12264018,  0.13487358, -0.40809375, -0.4061165 , -0.39266393,
            0.04690954, -0.10905049,  0.00406707,  0.00732412,  0.03587759])
    >>> pdf_clayton(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=False)
    array([0.88458189, 1.1443921 , 0.66491654, 0.66623254, 0.67525564,
           1.0480272 , 0.89668514, 1.00407535, 1.007351  , 1.03652896])
    """
    dcopula = np.empty_like(u_values[:, 0])
    sum_seq = 0.0
    d = float(u_values.shape[1])
    t = np.sum(ipsi_clayton(x=u_values, theta=theta), axis=1)
    lu = np.sum(np.log(u_values), axis=1)
    const_a = np.log1p(theta)
    const_b = 1.0 + theta
    const_c = np.log1p(-t)
    if theta > 0.0:
        sum_seq = np.array([i for i in np.arange(1.0, d)])
        sum_seq = np.sum(np.log1p(theta * sum_seq))
        const_c = np.log1p(t)
    const_d = d + 1.0 / theta
    for i in range(u_values.shape[0]):
        if theta < 0.0:
            if t[i] < 1.0:
                dcopula[i] = const_a - const_b * lu[i] - const_d * const_c[i]
            else:
                dcopula[i] = -np.Inf
        elif theta > 0.0:
            dcopula[i] = sum_seq - const_b * lu[i] - const_d * const_c[i]
        else:
            dcopula[i] = 0.0
        if not is_log:
            dcopula[i] = np.exp(dcopula[i])
    return dcopula

def diag_pdf_clayton(u_values, theta, is_log=False):
    """
    compute clayton copula diagonal pdf
    :param u_values:
    :param theta:
    :return:
    >>> diag_pdf_clayton(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=True)
    array([ 0.98084835, -0.87814578,  0.58088657,  0.12305888,  0.13268952,
            0.23601368,  0.86262679,  0.66752968, -0.04581703,  0.31014911])
    >>> diag_pdf_clayton(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=False)
    array([2.66671759, 0.41555273, 1.78762258, 1.13095101, 1.14189541,
           1.26619164, 2.36937639, 1.94941569, 0.95521672, 1.36362842])
    """
    y = diag_copula(u_values)
    d = float(u_values.shape[1])
    if is_log:
        return np.log(d) - (1.0 + 1.0 / theta) * np.log1p((d - 1.0) * \
               (1.0 - y ** theta))
    return d * (1.0 + (d - 1.0) * (1.0 - y ** theta)) ** (- (1.0 + 1.0 /
                                                              theta))

def copula_gumbel_cdf(u_values, theta):
    """
    compute franck copula cdf
    :param u_values:
    :param theta:
    :return:
    >>> copula_gumbel_cdf(np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ]),
    ...    10)
    array([0.63429604, 0.27406786, 0.45146423, 0.15136347, 0.05290861,
           0.48956205, 0.22830395, 0.05501846, 0.25932584, 0.24581602])
    """
    copula = np.empty_like(u_values[:, 0])
    for i in range(u_values.shape[0]):
        copula[i] = 0
        for j in range(u_values.shape[1]):
            copula[i] += (-np.log(float(u_values[i, j]))) ** (float(theta))
        copula[i] = np.exp(-((copula[i]) ** (1.0 / float(theta))))
    return copula


def log_copula_gumbel_cdf(u_values, theta):
    """
    compute franck copula cdf
    :param u_values:
    :param theta:
    :return:
    >>> log_copula_gumbel_cdf(np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ]),
    ...    0.1)
    array([-46736.81306738, -58889.15496375, -55271.78960881,  -6619.27103773,
           -59042.98546951,  -5657.15892143, -57597.91308742, -59047.18826512,
           -50986.12317908, -35519.51968965])
    """

    def s_j(j, alpha, d):
        if (alpha * float(j)).is_integer() or (
                alpha == 1.0 and float(int(j) == int(d))
        ):
            return (-1.0) ** (float(j) - np.ceil(alpha * float(j)))
        return 0.0

    def t_theta(ui_values, theta):
        results = 0.0
        for j in range(ui_values.shape[0]):
            results += np.exp(-float(ui_values[j]) ** (1.0 / float(theta)))
        return results

    def weird_notation(x, y, n):
        return factorial(float(x) * float(y)) / factorial(float(x) * float(y) -
                                                          float(n))

    def log_polyG(ui_values, theta):
        t_theta_i = t_theta(ui_values, theta)
        alpha = 1.0 / float(theta)
        logsum = np.empty_like(ui_values)
        for j in range(ui_values.shape[0]):
            logsum[j] = np.log(abs(weird_notation(
                alpha,
                j + 1,
                ui_values.shape[0]
            )))
            logsum[j] += float(j + 1) * np.log(float(t_theta_i))
            logsum[j] += float(t_theta_i)
            logsum[j] -= np.log(factorial(float(j + 1)))
            logsum[j] += np.log(poisson.cdf(int(ui_values.shape[0] - j + 1),
                                            float(t_theta_i)))
            logsum[j] = np.exp(logsum[j])
            logsum[j] *= s_j(j + 1, alpha, ui_values.shape[0])
        return lssum(logsum)

    lc_theta = np.empty_like(u_values[:, 0])
    for i in range(u_values.shape[0]):
        lc_theta[i] = float(u_values.shape[1]) * np.log(float(theta))
        lc_theta[i] -= t_theta(u_values[i, :], theta) ** (1.0 / float(theta))
        lc_theta[i] -= (1.0 / float(theta)) * np.log(t_theta(u_values[i, :],
                                                             theta))
        for j in range(u_values.shape[1]):
            lc_theta[i] += np.log((-np.log(u_values[i, j])) ** (float(theta) -
                                                                1.0))
            lc_theta[i] -= np.log(u_values[i, j])
        lc_theta[i] += log_polyG(u_values[i, :], theta)
    return lc_theta


def diag_copula(u_values):
    """
    compute theta for a gumbel copula with DMLE
    :param u_values:
    :return: diagonal copula
    >>> diag_copula(np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ]))
    array([0.72122885, 0.48840676, 0.63469281, 0.91851515, 0.34839029,
           0.99356872, 0.60683747, 0.30158193, 0.73040326, 0.83277053])
    """
    y = np.empty_like(u_values[:, 0])
    for i in range(u_values.shape[0]):
        y[i] = max(u_values[i, :])
    return y


def DMLE_copula_gumbel(u_values):
    """
    compute theta for a gumbel copula with DMLE
    :param u_values:
    :return: theta
    >>> DMLE_copula_gumbel(np.array([
    ...    [0.72122885, 0.64249391, 0.6771109 ],
    ...    [0.48840676, 0.36490127, 0.27721709],
    ...    [0.63469281, 0.4517949 , 0.62365817],
    ...    [0.87942847, 0.15136347, 0.91851515],
    ...    [0.34839029, 0.05604025, 0.08416331],
    ...    [0.48967318, 0.99356872, 0.66912132],
    ...    [0.60683747, 0.4841944 , 0.22833209],
    ...    [0.30158193, 0.26186022, 0.05502786],
    ...    [0.51942063, 0.73040326, 0.25935125],
    ...    [0.46365886, 0.2459    , 0.83277053]
    ...    ]))
    1.5136102146750419
    """
    theta = np.log(float(u_values.shape[0])) - lsum(
        -np.log(diag_copula(u_values))
    )
    theta = np.log(float(u_values.shape[1])) / theta
    return max([theta, 1.0])



def ipsi_frank(u_values, theta, is_log=False):
    """
    Compute iPsi function for Frank copula
    :param u_values:
    :param theta:
    :param is_log:
    :return:
    >>> ipsi_frank(np.array([
    ...   [0.42873569, 0.18285458, 0.9514195],
    ...   [0.25148149, 0.05617784, 0.3378213],
    ...   [0.79410993, 0.76175687, 0.0709562],
    ...   [0.02694249, 0.45788802, 0.6299574],
    ...   [0.39522060, 0.02189511, 0.6332237],
    ...   [0.66878367, 0.38075101, 0.5185625],
    ...   [0.90365653, 0.19654621, 0.6809525],
    ...   [0.28607729, 0.82713755, 0.7686878],
    ...   [0.22437343, 0.16907646, 0.5740400],
    ...   [0.66752741, 0.69487362, 0.3329266]
    ...   ]),
    ...   0.2)
    array([[0.791148  , 1.61895993, 0.04510005],
           [1.30709475, 2.78651154, 1.02049626],
           [0.21055968, 0.24900271, 2.55444583],
           [3.51840984, 0.72823612, 0.42610361],
           [0.86923904, 3.72534678, 0.42125181],
           [0.37009377, 0.90510924, 0.60976894],
           [0.09197708, 1.54811406, 0.35325138],
           [1.1816307 , 0.17302414, 0.24062066],
           [1.41846307, 1.69593035, 0.51357717],
           [0.37185113, 0.33437415, 1.03460728]])
    """
    def mapping_function(x):
        if x <= 0.01 * np.abs(theta):
            return - np.log(np.expm1(-x * theta) / np.expm1(-theta))
        else:
            if np.exp(-theta) > 0 and np.abs(theta - x * theta) < 1.0 / 2.0:
                return -np.log1p(np.exp(-theta) * np.expm1(theta - x * theta) /
                       np.expm1(-theta))
            else:
                return -np.log1p((np.exp(-x * theta) - np.exp(-theta)) /
                                 np.expm1(-theta))
    mapping_function = np.vectorize(mapping_function)
    if is_log:
        return np.log(mapping_function(u_values))
    return mapping_function(u_values)


def psi_frank(u_values, theta):
    """
    Compute Psi function for Frank copula
    :param u_values:
    :param theta:
    :param is_log:
    :return:
    >>> psi_frank(np.array([
    ...   [0.42873569, 0.18285458, 0.9514195],
    ...   [0.25148149, 0.05617784, 0.3378213],
    ...   [0.79410993, 0.76175687, 0.0709562],
    ...   [0.02694249, 0.45788802, 0.6299574],
    ...   [0.39522060, 0.02189511, 0.6332237],
    ...   [0.66878367, 0.38075101, 0.5185625],
    ...   [0.90365653, 0.19654621, 0.6809525],
    ...   [0.28607729, 0.82713755, 0.7686878],
    ...   [0.22437343, 0.16907646, 0.5740400],
    ...   [0.66752741, 0.69487362, 0.3329266]
    ...   ]),
    ...   0.2)
    array([[0.62819295, 0.81834625, 0.36287933],
           [0.75972015, 0.93988774, 0.69230893],
           [0.42741191, 0.44210586, 0.92474177],
           [0.97065876, 0.60899809, 0.50765389],
           [0.65105904, 0.97608253, 0.50591179],
           [0.48734711, 0.66120387, 0.57101593],
           [0.38132619, 0.80627025, 0.4811596 ],
           [0.73189809, 0.41293628, 0.4389142 ],
           [0.78231684, 0.83069672, 0.53847762],
           [0.48799062, 0.47418202, 0.69595363]])
    >>> psi_frank(np.array([
    ...   [0.42873569, 0.18285458, 0.9514195],
    ...   [0.25148149, 0.05617784, 0.3378213],
    ...   [0.79410993, 0.76175687, 0.0709562],
    ...   [0.02694249, 0.45788802, 0.6299574],
    ...   [0.39522060, 0.02189511, 0.6332237],
    ...   [0.66878367, 0.38075101, 0.5185625],
    ...   [0.90365653, 0.19654621, 0.6809525],
    ...   [0.28607729, 0.82713755, 0.7686878],
    ...   [0.22437343, 0.16907646, 0.5740400],
    ...   [0.66752741, 0.69487362, 0.3329266]
    ...   ]),
    ...   -40)
    array([[0.98928161, 0.99542864, 0.97621451],
           [0.99371296, 0.99859555, 0.99155447],
           [0.98014725, 0.98095608, 0.9982261 ],
           [0.99932644, 0.9885528 , 0.98425106],
           [0.99011948, 0.99945262, 0.98416941],
           [0.98328041, 0.99048122, 0.98703594],
           [0.97740859, 0.99508634, 0.98297619],
           [0.99284807, 0.97932156, 0.9807828 ],
           [0.99439066, 0.99577309, 0.985649  ],
           [0.98331181, 0.98262816, 0.99167684]])
    >>> psi_frank(np.array([
    ...   [0.42873569, 0.18285458, 0.9514195],
    ...   [0.25148149, 0.05617784, 0.3378213],
    ...   [0.79410993, 0.76175687, 0.0709562],
    ...   [0.02694249, 0.45788802, 0.6299574],
    ...   [0.39522060, 0.02189511, 0.6332237],
    ...   [0.66878367, 0.38075101, 0.5185625],
    ...   [0.90365653, 0.19654621, 0.6809525],
    ...   [0.28607729, 0.82713755, 0.7686878],
    ...   [0.22437343, 0.16907646, 0.5740400],
    ...   [0.66752741, 0.69487362, 0.3329266]
    ...   ]),
    ...   -10)
    array([[0.95712886, 0.98171545, 0.90486527],
           [0.97485315, 0.99438248, 0.96621969],
           [0.92059451, 0.9238295 , 0.99290471],
           [0.99730587, 0.95421383, 0.93700824],
           [0.96048014, 0.99781059, 0.93668164],
           [0.93312595, 0.961927  , 0.94814684],
           [0.90964101, 0.98034637, 0.93190918],
           [0.97139377, 0.91729209, 0.92313647],
           [0.9775638 , 0.98309319, 0.94259952],
           [0.93325157, 0.93051719, 0.96670913]])
    """
    if theta > 0.0:
        return -log1mexp(u_values - log1mexp(theta)) / theta
    elif theta == 0.0:
        return np.exp(-u_values)
    elif theta < np.log(np.finfo(float).eps):
        return -log1pexp(-(u_values + theta)) / theta
    return - np.log1p(np.exp(-u_values) * np.expm1(-theta)) / theta

def diag_pdf_frank(u_values, theta, is_log=False):
    """
    compute frank copula diagonal pdf
    :param u_values:
    :param theta:
    :return:
    >>> diag_pdf_frank(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=True)
    array([ 0.9904959 , -1.00142163,  0.6200179 ,  0.17221735,  0.18204756,
            0.28624782,  0.88191526,  0.70127801, -0.00374368,  0.35967931])
    >>> diag_pdf_frank(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=False)
    array([2.6925694 , 0.36735682, 1.85896133, 1.187936  , 1.19967124,
           1.33142237, 2.41552162, 2.01632796, 0.99626332, 1.43286983])
    """
    yt = diag_copula(u_values) * theta
    d = float(u_values.shape[1])

    def delt_dcom(x):
        ep = ((np.exp(-x) - np.exp(x - theta)) / (-np.expm1(-x)))
        delt = np.exp(-x) * (1.0 + ep)
        d1 = d - 1.0
        dcom = d + d1 * ep + (1.0 + ep) * delt
        return delt, dcom, d1

    def dDiagepoly2(x):
        delt, dcom, d1 = delt_dcom(x)
        res = d1 * (d - 2.0) / 2.0 * (1.0 + (d - 3.0) / 3.0 *
                                                    delt)
        res *= dcom
        if is_log:
            return np.log(d) - np.log(res)
        return d / res

    def dDiagepoly4(x):
        delt, dcom, d1 = delt_dcom(x)
        res = (d - 1.0) * (d - 2.0) / 2.0 * (1.0 + (d - 3.0) / 3.0 *
                                                   delt *
                                       (1 + (d - 4.0) / 4.0 * delt * (
                                           1.0 + (d - 5.0) / 5 * delt
                                       )))
        res *= dcom
        if is_log:
            return np.log(d) - np.log(res)
        return d / res

    def dDiagem1(x):
        h = -np.expm1(-theta)
        ie = -np.expm1(-x)
        res = (h / ie) ** (d - 1.0) - ie
        if is_log:
            return np.log(d) - x - np.log(res)
        return d * np.exp(-x) / res

    def mapping_function(x):
        if x > 25:
            return dDiagepoly2(x)
        elif x < 0.1:
            return dDiagem1(x)
        else:
            return dDiagepoly4(x)

    mapping_function = np.vectorize(mapping_function)
    return mapping_function(yt)

def pdf_frank(u_values, theta, is_log = False):
    """
    compute frank copula pdf
    :param u_values:
    :param theta:
    :return:
    >>> pdf_frank(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=False)
    array([0.94796045, 1.07458178, 0.91117583, 0.98067912, 0.99144689,
           0.9939432 , 0.94162409, 0.96927238, 1.02271257, 0.98591624])
    >>> pdf_frank(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=True)
    array([-0.05344249,  0.07193155, -0.09301939, -0.01950997, -0.00858989,
           -0.00607522, -0.06014914, -0.03120961,  0.02245848, -0.01418388])
    """
    copula = np.empty_like(u_values[:, 0])
    for i in range(u_values.shape[0]):
        copula[i] = (float(theta) /
                     (1.0 - np.exp(-float(theta)))
                     ) ** (float(u_values.shape[1]) - 1.0)
        h_res_i = (1.0 - np.exp(-float(theta))
                   ) ** (1.0 - float(u_values.shape[1]))
        sum_ui = 0.0
        for j in range(u_values.shape[1]):
            h_res_i *= (1.0 - np.exp(-float(theta) * float(u_values[i, j])))
            sum_ui += float(u_values[i, j])
        copula[i] *= polylog(-(float(u_values.shape[1]) - 1.0), h_res_i)
        copula[i] *= np.exp(-float(theta) * sum_ui) / h_res_i
    if is_log:
        return np.log(copula)
    return copula

def ipsi_gumbel(u_values, theta, is_log=False):
    """
    Compute iPsi function for gumbel copula
    :param u_values:
    :param theta:
    :param is_log:
    :return:
    >>> ipsi_gumbel(np.array([
    ...   [0.42873569, 0.18285458, 0.9514195],
    ...   [0.25148149, 0.05617784, 0.3378213],
    ...   [0.79410993, 0.76175687, 0.0709562],
    ...   [0.02694249, 0.45788802, 0.6299574],
    ...   [0.39522060, 0.02189511, 0.6332237],
    ...   [0.66878367, 0.38075101, 0.5185625],
    ...   [0.90365653, 0.19654621, 0.6809525],
    ...   [0.28607729, 0.82713755, 0.7686878],
    ...   [0.22437343, 0.16907646, 0.5740400],
    ...   [0.66752741, 0.69487362, 0.3329266]
    ...   ]),
    ...   1.2)
    array([[0.81923327, 1.88908593, 0.02733237],
           [1.47231458, 3.55739554, 1.10313864],
           [0.17190186, 0.20976237, 3.21401011],
           [4.67297101, 0.74347847, 0.3959922 ],
           [0.91460233, 4.99665702, 0.39068015],
           [0.33531508, 0.95887481, 0.60372092],
           [0.06408581, 1.79316182, 0.31736126],
           [1.30892336, 0.13611697, 0.20141246],
           [1.61947932, 1.99408403, 0.49340591],
           [0.33719654, 0.29741158, 1.1209654 ]])
    >>> ipsi_gumbel(np.array([
    ...   [0.42873569, 0.18285458, 0.9514195],
    ...   [0.25148149, 0.05617784, 0.3378213],
    ...   [0.79410993, 0.76175687, 0.0709562],
    ...   [0.02694249, 0.45788802, 0.6299574],
    ...   [0.39522060, 0.02189511, 0.6332237],
    ...   [0.66878367, 0.38075101, 0.5185625],
    ...   [0.90365653, 0.19654621, 0.6809525],
    ...   [0.28607729, 0.82713755, 0.7686878],
    ...   [0.22437343, 0.16907646, 0.5740400],
    ...   [0.66752741, 0.69487362, 0.3329266]
    ...   ]),
    ...   1.2, is_log=True)
    array([[-0.19938642,  0.63609307, -3.59968356],
           [ 0.38683571,  1.26902869,  0.09815943],
           [-1.76083155, -1.56177998,  1.16751941],
           [ 1.54179506, -0.29641547, -0.92636075],
           [-0.08926592,  1.60876909, -0.93986609],
           [-1.09268464, -0.04199476, -0.50464323],
           [-2.74753233,  0.58398044, -1.14771453],
           [ 0.26920494, -1.9942407 , -1.60240044],
           [ 0.48210469,  0.69018481, -0.70642309],
           [-1.08708931, -1.21263832,  0.11419028]])
    """
    if is_log:
        return theta * np.log(-np.log(u_values))
    return (-np.log(u_values)) ** theta


def psi_gumbel(u_values, theta):
    """
    Compute Psi function for Frank copula
    :param u_values:
    :param theta:
    :param is_log:
    :return:
    >>> psi_gumbel(np.array([
    ...   [0.42873569, 0.18285458, 0.9514195],
    ...   [0.25148149, 0.05617784, 0.3378213],
    ...   [0.79410993, 0.76175687, 0.0709562],
    ...   [0.02694249, 0.45788802, 0.6299574],
    ...   [0.39522060, 0.02189511, 0.6332237],
    ...   [0.66878367, 0.38075101, 0.5185625],
    ...   [0.90365653, 0.19654621, 0.6809525],
    ...   [0.28607729, 0.82713755, 0.7686878],
    ...   [0.22437343, 0.16907646, 0.5740400],
    ...   [0.66752741, 0.69487362, 0.3329266]
    ...   ]),
    ...   1.2)
    array([[0.61034427, 0.78449875, 0.38314216],
           [0.72866953, 0.91322228, 0.66711104],
           [0.43814072, 0.45063321, 0.89558443],
           [0.95198356, 0.59359729, 0.50641834],
           [0.63043035, 0.95944934, 0.50493238],
           [0.48911264, 0.63939466, 0.56071577],
           [0.39890033, 0.77277837, 0.48384529],
           [0.70297971, 0.42582847, 0.44791995],
           [0.74988568, 0.79662481, 0.53276337],
           [0.48966058, 0.47790784, 0.67038358]])
    """
    return np.exp(-u_values ** (1.0 / theta))

def diag_pdf_gumbel(u_values, theta, is_log=False):
    """
    compute frank copula diagonal pdf
    :param u_values:
    :param theta:
    :return:
    >>> diag_pdf_gumbel(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=True)
    array([  -6.55858673, -257.13458817,  -50.29601565, -106.33588414,
           -105.08436706,  -91.86224008,  -19.02297494,  -40.4347328 ,
           -128.83053864,  -82.60105914])
    >>> diag_pdf_gumbel(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    0.2,
    ...    is_log=False)
    array([1.41788815e-003, 2.12748865e-112, 1.43455743e-022, 6.59040780e-047,
           2.30377070e-046, 1.27272929e-040, 5.47553997e-009, 2.75054445e-018,
           1.12100608e-056, 1.33910865e-036])
    """
    y = diag_copula(u_values)
    d = float(u_values.shape[1])
    alpha = 1.0 / theta
    da = d ** alpha
    if is_log:
        return (da - 1.0) * np.log(y) + alpha * np.log(d)
    return da * y ** (da - 1.0)

def pdf_gumbel(u_values, theta, is_log = False):
    """
    compute frank copula pdf
    :param u_values:
    :param theta:
    :return:
    >>> pdf_gumbel(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    1.2,
    ...    is_log=False)
    array([0.62097606, 1.39603813, 0.58225969, 0.85072331, 0.88616848,
           1.10022557, 0.66461897, 0.33769735, 1.15561848, 1.01957628])
    >>> pdf_gumbel(np.array([
    ...    [0.42873569, 0.18285458, 0.9514195],
    ...    [0.25148149, 0.05617784, 0.3378213],
    ...    [0.79410993, 0.76175687, 0.0709562],
    ...    [0.02694249, 0.45788802, 0.6299574],
    ...    [0.39522060, 0.02189511, 0.6332237],
    ...    [0.66878367, 0.38075101, 0.5185625],
    ...    [0.90365653, 0.19654621, 0.6809525],
    ...    [0.28607729, 0.82713755, 0.7686878],
    ...    [0.22437343, 0.16907646, 0.5740400],
    ...    [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    1.2,
    ...    is_log=True)
    array([-0.47646275,  0.33363832, -0.54083873, -0.16166834, -0.12084819,
            0.09551522, -0.40854139, -1.08560521,  0.14463568,  0.01938713])
    """

    def s_j(j, alpha, d):
        assert 0.0 < alpha
        assert alpha <= 1.0
        assert d >= 0.0 and 0.0 <= float(j)
        if alpha == 1.0:
            if int(j) == int(d):
                return 1.0
            else:
                return (-1.0)**(d-float(j))
        else:
            x = alpha * float(j)
            if x != np.floor(x):
                return (-1.0)**(float(j)-np.ceil(x))
            else:
                return 0.0
        return np.NaN

    def log_polyG(lx, alpha, d):
        res = np.zeros(shape=int(d))
        x = np.exp(lx)
        for j in range(1, int(d) + 1):
            res[j-1] += np.log(abs(binom(alpha * float(j), d))) + np.log(
                factorial(d))
            res[j-1] += float(j) * lx
            res[j-1] += x - np.log(factorial(float(j)))
            res[j-1] += poisson.logcdf(d - float(j), x)
            res[j-1] = s_j(j, alpha, d) * np.exp(res[j-1])
        return lssum(res)

    d = float(u_values.shape[1])
    alpha = 1.0 / theta
    lip = ipsi_gumbel(u_values, theta, is_log=False)
    lnt = np.zeros(shape=u_values.shape[0])
    for i in range(u_values.shape[0]):
        lnt[i] = lsum(lip[i, :])
    mlu = -np.log(u_values)
    lmlu = np.log(mlu)
    lx = alpha * lnt
    ls = np.zeros(shape=u_values.shape[0])
    for i in range(u_values.shape[0]):
        ls[i] = log_polyG(lx[i], alpha, d) - d * lx[i] / alpha
    lnC = -np.exp(lx)
    dcopula = lnC + d * np.log(theta) + np.sum((theta - 1.0) * lmlu + mlu,
                                               axis=1) + ls
    if is_log:
        return dcopula
    return np.exp(dcopula)

def m_step_theta(u_values, k_states, params_list, copula):
    """
    wrapper function to compute lambda for the right copula
    :param u_values:
    :param k_states:
    :param params_list:
    :param copula:
    :return:
    """
    return eval("m_step_" + copula + "_lambda")(
        u_values,
        k_states,
        params_list[copula]
    )

def samic_delta(copula, params_list, threshold):
    """
    Return true if the difference between two iteration of samic if less than
    the threhsold
    :param copula: str with the copula name
    :param params_list: list of model parameters
    :param threshold: flood withe the minimal difference to reach
    :return: bool
    """
    return max([
        abs(params_list['pi'] - params_list['pi_old']),
        abs(params_list[copula]['theta'] - params_list[copula]['theta_old'])
    ]) >= threshold

def samic_e_k(u_values, copula, params_list, k_state):
    """
    compute proba for each line to be in one component or the other
    :param u_values:
    :param copula:
    :param params_list:
    :param k_values:
    :return:
    """
    copula_density = {
        'clayton': copula_clayton_pdf,
        'franck': copula_clayton_pdf,
        'gumbel': copula_gumbel_pdf
    }
    k_state = params_list['pi'] / (
            params_list['pi'] + (
                1.0 - params_list['pi']
            ) * copula_density[copula](
                u_values,
                params_list[copula]['theta']
            )
        )
    return k_state

def samic_mix(u_values, copula, theta, k_states):
    """
    pdf of the samic mixture for a given copula
    :param u_values:
    :param copula:
    :param params_list:
    :param k_states:
    :return:
    """
    copula_density = {
        'clayton': copula_clayton_pdf,
        'franck': copula_clayton_pdf,
        'gumbel': copula_gumbel_pdf
    }
    return lsum(np.log(k_states + (1.0 - k_states) * copula_density[copula](
        u_values,
        theta
    )))

def samic_min_theta(u_values, copula, k_state):
    """
    find theta that minimize the likelihood of the copula density
    :param u_values:
    :param copula:
    :param params_list:
    :return:
    """
    DMLE_copula = {
        'clayton': DMLE_copula_clayton,
        'franck': DMLE_copula_franck,
        'gumbel': DMLE_copula_gumbel
    }
    theta_DMLE = DMLE_copula[copula](u_values)
    return theta_DMLE
    res = minimize(
        fun=lambda x: samic_mix(u_values, copula, x, k_state),
        x0=theta_DMLE,
        constraints=[
            {'type': 'ineq', 'fun': lambda x: max([
                1e-14,
                theta_DMLE - 0.5
            ])},
            {'type': 'ineq', 'fun': lambda x: min([
                100,
                theta_DMLE + 0.5
            ])}
        ],
        options={'maxiter': 1000}
    )
    if res.success:
        return res.x[0]
    else:
        print(res)
        return np.NaN

def samic(x_score, threshold=1e-4, log_name=""):
    """
    implementation of the samic method for m samples
    :param x_score np.array of score (measures x samples)
    :param threshold float min delta between every parameters between two
    iterations
    :param log_name str name of the log files
    :return (theta: dict, lidr: list) with thata the model parameters and
    lidr the local idr values for each measures
    >>> THETA_TEST_0 = {'pi': 0.6, 'mu': 0.0, 'sigma': 1.0, 'rho': 0.0}
    >>> THETA_TEST_1 = {'pi': 0.6, 'mu': 4.0, 'sigma': 3.0, 'rho': 0.75}
    >>> THETA_TEST = {'pi': 0.2,
    ...               'mu': THETA_TEST_1['mu'] - THETA_TEST_0['mu'],
    ...               'sigma': THETA_TEST_0['sigma'] / THETA_TEST_1['sigma'],
    ...               'rho': 0.75}
    >>> DATA = sim_m_samples(n_value=1000,
    ...                      m_sample=2,
    ...                      theta_0=THETA_TEST_0,
    ...                      theta_1=THETA_TEST_1)
    >>> samic(DATA["X"], threshold=0.01)
    """
    u_values = compute_empirical_marginal_cdf(compute_rank(x_score))
    copula_list = ["clayton", "franck", "gumbel"]
    DMLE_copula = {
        'clayton': DMLE_copula_clayton,
        'franck': DMLE_copula_franck,
        'gumbel': DMLE_copula_gumbel
    }
    params_list = {
        'pi': 0.5,
        'pi_old': np.Inf
    }
    k_state = [0.0] * int(u_values.shape[0])
    for copula in copula_list:
        params_list[copula] = {
            'theta': DMLE_copula[copula](u_values),
            'theta_old': np.Inf
        }
        while samic_delta(copula, params_list, threshold):
            params_list['pi_old'] = params_list['pi']
            params_list[copula]['theta_old'] = params_list[copula]['theta']
            k_state = samic_e_k(
                u_values=u_values,
                copula=copula,
                params_list=params_list,
                k_state=k_state
            )
            params_list['pi'] = m_step_pi(
                k_state=k_state
            )
            params_list[copula]['theta'] = samic_min_theta(
                u_values=u_values,
                copula=copula,
                k_state=k_state,
            )
            print(params_list)
    return params_list


def pseudo_likelihood(x_score, threshold=0.0001, log_name=""):
    """
    pseudo likelhood optimization for the copula model parameters
    :param x_score np.array of score (measures x samples)
    :param threshold float min delta between every parameters between two
    iterations
    :param log_name str name of the log files
    :return (theta: dict, lidr: list) with thata the model parameters and
    lidr the local idr values for each measures
    >>> THETA_TEST_0 = {'pi': 0.6, 'mu': 0.0, 'sigma': 1.0, 'rho': 0.0}
    >>> THETA_TEST_1 = {'pi': 0.6, 'mu': 4.0, 'sigma': 3.0, 'rho': 0.75}
    >>> THETA_TEST = {'pi': 0.2,
    ...               'mu': THETA_TEST_1['mu'] - THETA_TEST_0['mu'],
    ...               'sigma': THETA_TEST_0['sigma'] / THETA_TEST_1['sigma'],
    ...               'rho': 0.75}
    >>> DATA = sim_m_samples(n_value=10000,
    ...                      m_sample=2,
    ...                      theta_0=THETA_TEST_0,
    ...                      theta_1=THETA_TEST_1)
    >>> # (THETA_RES, LIDR) = pseudo_likelihood(DATA["X"],
    # ...                                      threshold=0.01)
    """
    theta_t0 = deepcopy(THETA_INIT)
    theta_t1 = deepcopy(THETA_INIT)
    k_state = [0.0] * int(x_score.shape[0])
    lidr = [0.0] * int(x_score.shape[0])
    logger = {
        'logl': list(),
        'pi': list(),
        'mu': list(),
        'sigma': list(),
        'rho': list(),
        'pseudo_data': list()
    }
    logl_t1 = -np.inf
    u_values = compute_empirical_marginal_cdf(compute_rank(x_score))
    while delta(theta_t0, theta_t1, threshold, logl_t1):
        del theta_t0
        theta_t0 = deepcopy(theta_t1)
        z_values = compute_z_from_u(u_values=u_values,
                                    theta=theta_t1)
        (theta_t1, k_state, logger) = em_pseudo_data(
            z_values=z_values,
            logger=logger,
            k_state=k_state,
            theta=theta_t1,
            threshold=threshold
        )
        lidr = local_idr(
            z_values=z_values,
            theta=theta_t1
        )
        logl_t1 = loglikelihood(
            z_values=z_values,
            k_state=k_state,
            theta=theta_t1
        )
        logger = log.add_log(
            log=logger,
            theta=theta_t1,
            logl=logl_t1,
            pseudo=True
        )
    log.plot_log(logger, str(log_name) + "_log.pdf")
    log.plot_classif(
        x_score,
        u_values,
        z_values,
        lidr,
        str(log_name) + "_classif.pdf"
    )
    log.LOGGER.debug("%s", str(theta_t1))
    return (theta_t1, lidr)


THETA_INIT = {
    'pi': 0.5,
    'mu': -1.0,
    'sigma': 1.0,
    'rho': 0.9
}

if __name__ == "__main__":
    THETA_TEST_0 = {'pi': 0.6, 'mu': 0.0, 'sigma': 1.0, 'rho': 0.0}
    THETA_TEST_1 = {'pi': 0.6, 'mu': 4.0, 'sigma': 3.0, 'rho': 0.75}
    THETA_TEST = {'pi': 0.2,
                  'mu': THETA_TEST_1['mu'] - THETA_TEST_0['mu'],
                  'sigma': THETA_TEST_0['sigma'] / THETA_TEST_1['sigma'],
                  'rho': 0.75}
    DATA = sim_m_samples(n_value=1000,
                         m_sample=2,
                         theta_0=THETA_TEST_0,
                         theta_1=THETA_TEST_1)
    samic(DATA["X"], threshold=0.01)
    import doctest

    doctest.testmod()
