#!/usr/bin/python3

"""
Package to compute IDR from n replicates
"""

import math
from copy import deepcopy
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
#  from scipy.optimize import brentq
import numpy as np
import pandas as pd
from pynverse import inversefunc
import pylab

MU_1_MIN = 0.0
MU_1_MAX = 20.0
SIGMA_1_MIN = 0.2
SIGMA_1_MAX = 20.0
RHO_1_MIN = 0.10
RHO_1_MAX = 0.99
PI_1_MIN = 0.01
PI_1_MAX = 0.99
THETA_INIT = {'pi': 0.5,
              'mu': 4.0,
              'sigma': 1.0,
              'rho': 0.5}


def read_peak(file_name):
    """
    read peak file
    """
    return file_name


def cov_matrix(m_sample, theta):
    """
    compute multivariate_normal covariance matrix
    """
    cov = np.full(shape=(int(m_sample), int(m_sample)),
                  fill_value=float(theta['rho']) * float(theta['sigma']))
    np.fill_diagonal(a=cov,
                     val=float(theta['sigma']))
    return cov


def sim_multivariate_gaussian(n_value, m_sample, theta):
    """
    draw from a multivariate Gaussian distribution
    """
    cov = cov_matrix(m_sample=m_sample,
                     theta=theta)
    return np.random.multivariate_normal(mean=[float(theta['mu'])] *
                                         int(m_sample),
                                         cov=cov,
                                         size=int(n_value))


def sim_m_samples(n_value, m_sample, theta):
    """
    simulate sample where position score are drawn from two different
    multivariate Gaussian distribution
    """
    scores = sim_multivariate_gaussian(n_value=n_value,
                                       m_sample=m_sample,
                                       theta=theta)
    spurious = sim_multivariate_gaussian(n_value=n_value,
                                         m_sample=m_sample,
                                         theta={'mu': 0,
                                                'sigma': 1,
                                                'rho': 0})
    k_state = list()
    for i in range(int(n_value)):
        k_state.append(True)
        if not bool(bernoulli.rvs(p=theta['pi'], size=1)):
            scores[i] = spurious[i]
            k_state[i] = False
    return {'X': scores, 'K': k_state}


def compute_rank(x_score):
    """
    transform x a n*m matrix of score into an n*m matrix of rank ordered by
    row.
    """
    rank = np.empty_like(x_score)
    for j in range(x_score.shape[1]):
        # we want the rank to start at 1
        rank[:, j] = rankdata(x_score[:, j])
    return rank


def compute_empirical_marginal_cdf(rank):
    """
    normalize ranks to compute empirical marginal cdf and scale by n / (n+1)
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
    """
    # fixed g function for given theta
    g_f = lambda x_values: g_function(z_values=x_values, theta=theta)
    # inverse of g functon for given theta
    g_m1 = lambda x_values: inversefunc(g_f,
                                        y_values=x_values,
                                        image=[0, 1],
                                        open_domain=False,
                                        domain=[min([-3.0, theta['mu'] - 3.0]),
                                                max([3.0, theta['mu'] + 3.0])])
    z_values = np.empty_like(u_values)
    for i in range(u_values.shape[0]):
        for j in range(u_values.shape[1]):
            z_values[i][j] = g_m1(u_values[i][j])
    print(theta)
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
        print("error: h_function: " + str(err))
        print(cov)
        print(theta)
    return pd.Series(x_values)


def e_step_k(z_values, k_state, theta):
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
    return (1.0 / (nb_non_diag * theta['sigma'] * float(sum(k_state)))) *\
        z_norm_time


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
        print("error: logLikelihood: " + str(err))
        print(h1_x[i])
        print(theta)
        quit(-1)


def delta(theta_t0, theta_t1, threshold, logl):
    """
    compute the maximal variation between t0 and t1 for the estimated
    parameters
    """
    esp = 0
    for parameters in theta_t0:
        if abs(theta_t0[parameters] - theta_t1[parameters]) > threshold:
            esp += 1
    if esp == 0 and logl != -np.inf:
        return False
    return True


def em_pseudo_data(z_values,
                   log,
                   theta,
                   k_state,
                   threshold=0.001):
    """
    EM optimization of theta for pseudo-data
    """
    theta_t1 = deepcopy(theta)
    logl = 0.0
    logl_t1 = -np.inf
    while delta(theta, theta_t1, threshold, logl_t1):
        logl = logl_t1
        theta = deepcopy(theta_t1)
        k_state = e_step_k(z_values=z_values,
                           k_state=k_state,
                           theta=theta)
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
        log = add_log(log, theta_t1, logl_t1)
        if logl_t1 - logl < 0.0:
            print("warning: EM decreassing logLikelihood: " +
                  str(logl_t1 - logl))
            print(theta_t1)
    return (theta_t1, k_state, log)


def pseudo_likelihood(x_score, threshold=0.001):
    """
    pseudo likelhood optimization for the copula model parameters
    """
    theta = THETA_INIT
    theta_t1 = deepcopy(theta)
    logl = 0.0
    k_state = [0.0] * int(x_score.shape[0])
    u_values = [0.0] * int(x_score.shape[0])
    z_values = [0.0] * int(x_score.shape[0])
    lidr = [0.0] * int(x_score.shape[0])
    u_values = compute_empirical_marginal_cdf(compute_rank(x_score))
    log = {'logl': list(),
           'pi': list(),
           'mu': list(),
           'sigma': list(),
           'rho': list()}
    logl = 0.0
    logl_t1 = -np.inf
    while delta(theta, theta_t1, threshold, logl_t1):
        logl = logl_t1
        theta = deepcopy(theta_t1)
        z_values = compute_z_from_u(u_values=u_values,
                                    theta=theta)
        (theta_t1, k, log) = em_pseudo_data(z_values=z_values,
                                            log=log,
                                            k_state=k_state,
                                            theta=theta,
                                            threshold=threshold)
        lidr = local_idr(z_values=z_values,
                         lidr=lidr,
                         theta=theta_t1)
        logl_t1 = loglikelihood(z_values=z_values,
                                k_state=k_state,
                                theta=theta_t1)
        log = add_log(log,
                      theta_t1,
                      logl_t1)
        if logl_t1 - logl < 0.0:
            print("warning: pseudo data decreassing logLikelihood: " +
                  str(logl_t1 - logl))
            print(theta)
        plot_log(log, "log.pdf")
        plot_classif(x_score,
                     u_values,
                     z_values,
                     lidr,
                     "classif.pdf")
    print(theta)
    return (theta, lidr, k)


def add_log(log, theta, logl):
    """
    function to append thata and ll value to the logs
    """
    log['logl'].append(logl)
    for parameters in theta:
        log[parameters].append(theta[parameters])
    return log


def plot_log(log, file_name):
    """
    plot logs into a file
    """
    pylab.subplot(5, 1, 1)
    pylab.plot(np.linspace(start=0,
                           stop=len(log['logl']),
                           num=len(log['logl'])),
               log['logl'])
    pylab.ylabel('logLikelihood')
    pylab.subplot(5, 1, 2)
    pylab.plot(np.linspace(start=0,
                           stop=len(log['logl']),
                           num=len(log['logl'])),
               log['pi'])
    pylab.ylabel('pi')
    pylab.subplot(5, 1, 3)
    pylab.plot(np.linspace(start=0,
                           stop=len(log['logl']),
                           num=len(log['logl'])),
               log['mu'])
    pylab.ylabel('mu')
    pylab.subplot(5, 1, 4)
    pylab.plot(np.linspace(start=0,
                           stop=len(log['logl']),
                           num=len(log['logl'])),
               log['sigma'])
    pylab.ylabel('sigma')
    pylab.subplot(5, 1, 5)
    pylab.plot(np.linspace(start=0,
                           stop=len(log['logl']),
                           num=len(log['logl'])),
               log['rho'])
    pylab.ylabel('rho')
    pylab.savefig(file_name)


def plot_classif(x_score, u_values, z_values, lidr, file_name):
    """
    plot logs into a file
    """
    pylab.subplot(4, 1, 1)
    pylab.hist(x_score[:, 0], bins=1000, label=str(0))
    pylab.subplot(4, 1, 2)
    pylab.hist(z_values[:, 0], bins=1000, label=str(0))
    pylab.subplot(4, 1, 3)
    pylab.scatter(x_score[:, 1], z_values[:, 0], c=lidr)
    pylab.subplot(4, 1, 4)
    pylab.scatter(u_values[:, 1], z_values[:, 0], c=lidr)
    pylab.savefig(file_name)


THETA_TEST = {'pi': 0.2, 'mu': 3.0, 'sigma': 2.0, 'rho': 0.65}

DATA = sim_m_samples(n_value=10000,
                     m_sample=2,
                     theta=THETA_TEST)
(THETA_RES, LIDR, K) = pseudo_likelihood(DATA["X"], threshold=0.01)

pylab.plot(DATA['K'], K)
pylab.ylabel('k')
pylab.savefig("k_vs_estK.pdf")
pylab.plot(DATA['K'], LIDR)
pylab.ylabel('lidf')
pylab.savefig("k_vs_idr.pdf")
