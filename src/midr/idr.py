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
import numpy as np
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
    cov = cov_matrix(
        m_sample=m_sample,
        theta=theta
    )
    return np.random.multivariate_normal(
        mean=[float(theta['mu'])] * int(m_sample),
        cov=cov,
        size=int(n_value)
    )


def sim_m_samples(n_value, m_sample,
                  theta_0,
                  theta_1):
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
    scaling_factor = 0.99
    for i in range(int(n_value)):
        for j in range(int(m_sample)):
            x_score[i][j] = (1.0 - (float(rank[i][j] - 1) / n_value)) * \
                            scaling_factor
    return x_score


def g_function(z_values, theta):
    """
    compute scalded Gaussian cdf for Copula
    """
    f_pi = float(theta['pi'])
    return f_pi * norm.cdf(
        float(z_values),
        loc=float(theta['mu']),
        scale=np.sqrt(float(theta['sigma']))) + (1.0 - f_pi) * norm.cdf(
        float(z_values),
        loc=0.0,
        scale=1.0
    )


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
    :param grid:
    :param thread_num
    :return: list of z_value

    >>> z_from_u(
    ...    u_values=np.array([0.2, 0.3, 0.5, 0.9]),
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
    z_values = np.zeros(u_values.shape)
    if thread_num == 0:
        for i in range(u_values.shape[0]):
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
        x_values = multivariate_normal.pdf(
            x=z_values,
            mean=[float(theta['mu'])] * int(m_sample),
            cov=cov
        )
        return pd.Series(x_values)
    except ValueError as err:
        log.logging.exception("%s", "error: h_function: " + str(err))
        log.logging.exception("%s", str(cov))
        log.logging.exception("%s", str(theta))


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
    return np.array(lidr)


def m_step_pi(k_state, threshold):
    """
    compute maximization of pi
    """
    pi = float(sum(k_state)) / float(len(k_state))
    if 1.0 - pi <= threshold:
        log.logging.warning(
            "%s",
            "warning: pi maximization, empty reproducible group"
        )
        return 1.0 - threshold
    return pi


def m_step_alpha(l_state):
    """
    compute maximization of pi
    """
    return float(np.sum(l_state, axis=0)) / float(l_state.shape[0])


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
        log.logging.exception("%s", "error: logLikelihood: " + str(err))
        log.logging.exception("%s", str(h1_x[i]))
        log.logging.exception("%s", str(theta))
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
    >>> THETA_0 = {'pi': 0.2,
    ...            'mu': 0.0,
    ...            'sigma': 1.0,
    ...            'rho': 0.0}
    >>> DATA = sim_m_samples(n_value=1000,
    ...                      m_sample=2,
    ...                      theta_0=THETA_0,
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
            k_state=k_state,
            threshold=threshold
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
            log.logging.debug(
                "%s",
                "warning: EM decreassing logLikelihood rho: " +
                str(logl_t1 - logl_t0)
            )
            log.logging.debug("%s", str(theta_t1))
            return theta_t0, k_state, logger
        logger = log.add_log(
            log=logger,
            theta=theta_t1,
            logl=logl_t1,
            pseudo=False
        )
    return theta_t1, k_state, logger


def pseudo_likelihood(x_score, threshold=0.0001):
    """
    pseudo likelhood optimization for the copula model parameters
    :param x_score np.array of score (measures x samples)
    :param threshold float min delta between every parameters between two
    iterations
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
    >>> lidr = pseudo_likelihood(DATA["X"])
    >>> np.sum((np.array(lidr) < 0.5).all() == DATA["K"]) / len(lidr)
    """
    log.logging.info("%s", "computing idr")
    theta_t0 = deepcopy(THETA_INIT)
    theta_t1 = deepcopy(THETA_INIT)
    k_state = [0.0] * int(x_score.shape[0])
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
    z_values = u_values
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
        log.logging.info("%s", log_idr(theta_t1))
    return local_idr(
        z_values=z_values,
        theta=theta_t1
    )


def log_idr(theta):
    """
    return str of pseudo_likelihood parameter estimate
    :param theta:
    :return:
    """
    return str(
        '{' +
        '"pi": ' + str(theta['pi']) + ', ' +
        '"mu": ' + str(theta['mu']) + ', ' +
        '"sigma": ' + str(theta['sigma']) + ', ' +
        '"rho": ' + str(theta['rho']) +
        '}'
    )


THETA_INIT = {
    'pi': 0.5,
    'mu': -1.0,
    'sigma': 1.0,
    'rho': 0.9
}

if __name__ == "__main__":
    import doctest
    doctest.testmod()
