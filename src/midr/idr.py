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
from scipy.optimize import brentq
from scipy.special import factorial
from scipy.special import comb
import numpy as np
from mpmath import polylog
import pandas as pd
import midr.log as log


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


def lsum(x_values):
    """
    compute log sum_i x_i
    :param x_values:
    :return:
    """
    b_max = np.log(float(max(x_values)))
    results = 0.0
    for i in range(x_values.shape[0]):
        results += np.exp(np.log(float(x_values[i])) - b_max)
    return b_max + np.log(results)

def lssum(x_values):
    """
    compute log sum_i x_i with sign
    :param x_values:
    :return:
    """
    b_max = np.log(float(max(abs(x_values))))
    results = 0.0
    for i in range(x_values.shape[0]):
        if x_values[i] >= 0:
            results += np.exp(np.log(float(abs(x_values[i]))) - b_max)
        else:
            results -= np.exp(np.log(float(abs(x_values[i]))) - b_max)
    return b_max + np.log(results)

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


def copula_franck_pdf(u_values, theta):
    """
    compute franck copula pdf
    :param u_values:
    :param theta:
    :return:
    >>> copula_franck_pdf(np.array([
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
    ...    100)
    array([0.00000000e+00, 2.09407720e-09, 0.00000000e+00, 2.31523245e-61,
           2.02527721e-10, 0.00000000e+00, 5.63366190e-24, 4.07213022e-16,
           3.53860163e-28, 2.27206121e-31])
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
    return copula


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


def copula_clayton_pdf(u_values, theta):
    """
    compute franck copula cdf
    :param u_values:
    :param theta:
    :return:
    >>> copula_clayton_pdf(np.array([
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
    array([7.91969006e-15, 2.37733427e-15, 3.81218182e-15, 1.86177479e-14,
           5.36692925e-13, 1.00064000e-14, 3.32076461e-15, 7.62699923e-14,
           3.81082599e-15, 4.38314117e-15])
    """
    copula = np.empty_like(u_values[:, 0])
    alpha = 1.0 / float(theta)

    def t_theta(u_value, theta):
        """
        compute the sum of generator^-1
        :param u_value:
        :param theta:
        :return:
        """
        generator_1 = 0.0
        for j in range(u_value.shape[0]):
            generator_1 += (1.0 + float(u_value[j]) *
                            float(theta)) ** (-1.0 / float(theta))
        return generator_1
    for i in range(u_values.shape[0]):
        copula[i] = 1.0
        for k in range(u_values.shape[1]):
            copula[i] *= (float(theta) * float(k) + 1.0)
            copula_tmp = 1.0
            for j in range(u_values.shape[1]):
                copula_tmp *= float(u_values[i, j])
            copula[i] *= copula_tmp ** (- (1.0 + float(theta)))
            copula[i] *= (1.0 + t_theta(u_values[i, :], theta)
                          ) ** (-(u_values.shape[1] + alpha))
    return copula


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


def copula_gumbel_pdf(u_values, theta):
    """
    compute franck copula cdf
    :param u_values:
    :param theta:
    :return:
    >>> copula_gumbel_pdf(np.array([
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
    array([7.42521510e-08, 3.64657594e+04, 2.65790744e-03, 2.09134412e-11,
           1.35612755e+14, 3.18333298e-21, 1.95157053e+02, 7.28167691e+11,
           3.73452995e-01, 1.78052724e-02])
    """
    copula = np.empty_like(u_values[:, 0])
    alpha = 1.0 / float(theta)

    def t_theta(u_value, theta):
        """
        compute the sum of generator^-1
        :param u_value:
        :param theta:
        :return:
        """
        generator_1 = 0.0
        for j in range(u_value.shape[0]):
            generator_1 += np.exp(-float(u_value[j]) ** (1.0 / float(theta)))
        return generator_1

    def a_g(alpha, d, k):
        ag_res = factorial(float(d)) / factorial(float(k))
        for j in range(k):
            ag_res += float(
                comb(float(k), float(j))
            ) * float(
                comb(alpha * float(j), float(d))
            ) * ((-1.0) ** (float(d) - float(j)))
        return ag_res

    def p_g(x, d, alpha):
        pg_res = 0
        for k in range(d):
            pg_res += a_g(alpha, d, k) * (float(x) ** float(k))
        return pg_res

    for i in range(u_values.shape[0]):
        copula[i] = float(theta) ** u_values.shape[1]
        t_theta_i = t_theta(u_values[i, :], theta)
        copula[i] *= np.exp(-t_theta_i ** alpha)
        copula_num = 1.0
        copula_denum = t_theta_i ** float(u_values.shape[1])
        for j in range(u_values.shape[1]):
            copula_num *= (- np.log(u_values[i, j])) ** (float(theta) - 1)
            copula_denum *= u_values[i, j]
        copula[i] *= (copula_num / copula_denum)
        copula[i] *= p_g(t_theta_i, u_values.shape[1], alpha)
    return copula


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


def compute_diagonal(u_values):
    y_values = np.empty_like(u_values[:, 0])
    for i in range(u_values.shape[0]):
        y_values[i] = max(u_values[i, :])
    return y_values


def m_step_clayton_theta(u_values, k_states, params):


def m_step_grumbel_theta(u_values, k_states, params):
    y_values = compute_diagonal(u_values)
    sum_y_values = 0.0
    for i in range(y_values.shape[0]):
        sum_y_values += np.log(y_values[i])
    theta = float(np.log(u_values.shape[1])) / (
        np.log(u_values.shape[0]) - np.log(
            sum_y_values
        )
    )
    return max(theta, 1.0)


def samic(x_score, threshold=0.0001, log_name=""):
    """
    implementation of the samic method for m samples
    :param x_score np.array of score (measures x samples)
    :param threshold float min delta between every parameters between two
    iterations
    :param log_name str name of the log files
    :return (theta: dict, lidr: list) with thata the model parameters and
    lidr the local idr values for each measures
    """
    u_values = compute_empirical_marginal_cdf(compute_rank(x_score))
    copula_list = ["clayton", "franck", "gumbel"]
    params_list = {}
    for copula in copula_list:
        params_list[copula] = {'pi': 0, 'lambda': 0}
        k_state = [0.0] * int(x_score.shape[0])
        lidr = [0.0] * int(x_score.shape[0])
        params_old = {'pi': np.Inf, 'lambda': np.Inf, 'k': np.Inf}
        is_change = True
        while is_change:
            params_list['pi'] = m_step_pi(k_state)
            params_list['lambda'] = m_step_lambda(
                u_values,
                k_state,
                params_list,
                copula,
            )
            delta = max([
                abs(params_list[copula]['pi'] - params_old['pi']),
                abs(params_list[copula]['lambda'] - params_old['lambda'])
            ])
            if (delta < threshold):
                is_change = False
            else:
                params_old = params_list[copula]


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
    import doctest

    doctest.testmod()
