#!/usr/bin/python3

"""
Package to compute IDR from n replicates
"""

import math
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
#  from scipy.optimize import brentq
from copy import deepcopy
import numpy as np
import pandas as pd
from pynverse import inversefunc
#  import matplotlib.pyplot as plt
import pylab
import os
os.environ["DISPLAY"] = ":0"
import matplotlib
matplotlib.use('tkagg')

MU_1_MIN = 0.0
MU_1_MAX = 20.0
SIGMA_1_MIN = 0.2
SIGMA_1_MAX = 20.0
RHO_1_MIN = 0.10
RHO_1_MAX = 0.99
PI_1_MIN = 0.01
PI_1_MAX = 0.99
THETA_INIT = {'pi_1': 0.5,
              'mu_1': 4.0,
              'sigma_1': 1.0,
              'rho_1': 0.5}


def read_peak(file_name):
    """
    read peak file
    """
    return file_name


def cov_matrix(m, sigma_sq, rho):
    """
    compute multivariate_normal covariance matrix
    """
    cov = np.full(shape=(int(m), int(m)),
                  fill_value=float(rho) * float(sigma_sq))
    np.fill_diagonal(a=cov,
                     val=float(sigma_sq))
    return cov


def sim_multivariate_gaussian(n, m, mu, sigma_sq, rho):
    """
    draw from a multivariate Gaussian distribution
    """
    cov = cov_matrix(m=m,
                     sigma_sq=sigma_sq,
                     rho=rho)
    x = np.random.multivariate_normal(mean=[float(mu)] * int(m),
                                      cov=cov,
                                      size=int(n))
    return x


def sim_m_samples(n, m, mu, sigma_sq, rho, pi):
    """
    simulate sample where position score are drawn from two different
    multivariate Gaussian distribution
    """
    x = sim_multivariate_gaussian(n=n,
                                  m=m,
                                  mu=mu,
                                  sigma_sq=sigma_sq,
                                  rho=rho)
    spurious = sim_multivariate_gaussian(n=n,
                                         m=m,
                                         mu=0,
                                         sigma_sq=1,
                                         rho=0)
    k = list()
    for i in range(int(n)):
        k.append(True)
        if not bool(bernoulli.rvs(p=pi, size=1)):
            x[i] = spurious[i]
            k[i] = False
    return {'X': x, 'K': k}


def compute_rank(x):
    """
    transform x a n*m matrix of score into an n*m matrix of rank ordered by
    row.
    """
    r = np.empty_like(x)
    for j in range(x.shape[1]):
        # we want the rank to start at 1
        r[:, j] = rankdata(x[:, j])
    return r


def compute_empirical_marginal_cdf(r):
    """
    normalize ranks to compute empirical marginal cdf and scale by n / (n+1)
    """
    x = np.empty_like(r)
    n = float(r.shape[0])
    m = float(r.shape[1])
    scaling_factor = n / (n + 1.0)
    for i in range(int(n)):
        for j in range(int(m)):
            x[i][j] = (float(r[i][j]) / n) * scaling_factor
    return x


def G_function(z, theta):
    """
    compute scalded Gaussian cdf for Copula
    """
    sigma_1 = np.sqrt(float(theta['sigma_1']))
    z_norm = (float(z) - float(theta['mu_1'])) / sigma_1
    pi_1 = float(theta['pi_1']) / sigma_1
    u = pi_1 * norm.cdf(float(z_norm), loc=0, scale=1) + \
        (1.0 - pi_1) * norm.cdf(float(z), loc=0, scale=1)
    return u


def compute_z_from_u(u, k, theta):
    """
    compute u_ij from z_ij via the G_j function
    """
    G = lambda x: G_function(x,
                             theta=theta)
    #  G_m1 = lambda r: brentq(f=lambda x: G(x) - r,
    #                          a=min([-3, theta['mu_1'] - 3]),
    #                          b=max([3, theta['mu_1'] + 3]),
    #                          maxiter=1000)
    G_m1 = lambda r: inversefunc(G,
                                 y_values=r,
                                 image=[0, 1],
                                 open_domain=False,
                                 domain=[min([-3, theta['mu_1'] - 3]),
                                         max([3, theta['mu_1'] + 3])])
    z = np.empty_like(u)
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            z[i][j] = G_m1(u[i][j])
    ## plots
    #  pylab.hist(z[:, 0], bins=100, label=str(0))
    #  pylab.subplot(2, 1, 2)
    #  #  x = np.linspace(start=min(z[:, 0]), stop=max(z[:, 0]), num=1000)
    #  #  pylab.plot(x, x, label=str(0))
    #  #  y = np.empty_like(x)
    #  #  for i in range(len(y)):
    #  #      y[i] = G(x[i])
    #  #  pylab.plot(x, y, label=str(0))
    #  x = np.linspace(start=0.0, stop=1.0, num=1000)
    #  y = np.empty_like(x)
    #  for i in range(len(y)):
    #      y[i] = inv_cdf(x[i])
    #  pylab.plot(x, y, label=str(0))
    #  pylab.scatter(u[:, 0], z[:, 0], label=str(0), c=k)
    print(theta)
    #  pylab.show(block=True)
    #  # end plots
    return z


def h_function(z, m, mu, sigma_sq, rho):
    """
    compute the pdf of h0 or h1
    """
    try:
        cov = cov_matrix(m=int(m), sigma_sq=float(sigma_sq), rho=float(rho))
        x = multivariate_normal.pdf(x=z,
                                    mean=[float(mu)] * int(m),
                                    cov=cov)
    except ValueError as e:
        print("error: h_function: " + str(e))
        print(cov)
        print((mu, sigma_sq, rho))
    except Exception as e:
        print("error: h_function: " + str(e))
        print(cov)
        print((mu, sigma_sq, rho))
    return pd.Series(x)


def E_step_K(z, k, theta):
    """
    compute expectation of Ki
    """
    h0_x = h_function(z=z,
                      m=z.shape[1],
                      mu=0,
                      sigma_sq=1,
                      rho=0)
    h0_x = (1.0 - float(theta['pi_1'])) * h0_x
    h1_x = h_function(z=z,
                      m=z.shape[1],
                      mu=theta['mu_1'],
                      sigma_sq=theta['sigma_1'],
                      rho=theta['rho_1'])
    h1_x = float(theta['pi_1']) * h1_x
    k = h1_x / (h1_x + h0_x)
    return k.to_list()

def local_idr(z, lidr, theta):
    """
    compute local IDR
    """
    h0_x = h_function(z=z,
                      m=z.shape[1],
                      mu=0,
                      sigma_sq=1,
                      rho=0)
    h0_x = (1.0 - float(theta['pi_1'])) * h0_x
    h1_x = h_function(z=z,
                      m=z.shape[1],
                      mu=theta['mu_1'],
                      sigma_sq=theta['sigma_1'],
                      rho=theta['rho_1'])
    h1_x = float(theta['pi_1']) * h1_x
    lidr = h0_x / (h1_x + h0_x)
    return lidr.to_list()


def M_step_pi_1(k):
    """
    compute maximization of pi_1
    """
    pi_1 = float(sum(k)) / float(len(k))
    #  if pi_1 < PI_1_MIN:
    #      pi_1 = PI_1_MIN
    #  if pi_1 > PI_1_MAX:
    #      pi_1 = PI_1_MAX
    return pi_1


def M_step_mu_1(z, k):
    """
    compute maximization of mu_1
    0 < mu_1
    """
    denominator = float(z.shape[1]) * float(sum(k))
    numerator = 0.0
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            numerator += float(k[i]) * float(z[i][j])
    mu_1 = numerator / denominator
    #  if mu_1 < MU_1_MIN:
    #      mu_1 = MU_1_MIN
    #  if mu_1 > MU_1_MAX:
    #      mu_1 = MU_1_MAX
    return mu_1


def M_step_sigma_1(z, k, theta):
    """
    compute maximization of sigma_1
    """
    z_norm_sq = 0.0
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            z_norm_sq += float(k[i]) * (float(z[i][j]) -
                                        float(theta['mu_1']))**2
    sigma_1 = (1.0/ (float(z.shape[1]) * float(sum(k)))) * z_norm_sq
    #  if sigma_1 < SIGMA_1_MIN:
    #      sigma_1 = SIGMA_1_MIN
    #  if sigma_1 > SIGMA_1_MAX:
    #      sigma_1 = SIGMA_1_MAX
    return sigma_1


def M_step_rho_1(z, k, theta):
    """
    compute maximization of rho_1
    0 < rho_1 <= 1
    """
    nb_non_diag = float(z.shape[1])**2 - float(z.shape[1])
    z_norm_time = 0.0
    z_norm_time_i = 0.0
    for i in range(z.shape[0]):
        z_norm_time_i = 0.0
        for j in range(z.shape[1]):
            for l in range(z.shape[1]):
                if not l == j:
                    z_norm_time_i += (float(z[i][j]) - float(theta['mu_1'])) *\
                        (float(z[i][l]) - float(theta['mu_1']))
        z_norm_time += float(k[i]) * z_norm_time_i
    rho_1 = (1.0/ (nb_non_diag * theta['sigma_1'] * float(sum(k)))) * \
        z_norm_time
    #  if rho_1 < RHO_1_MIN:
    #      rho_1 = RHO_1_MIN
    #  if rho_1 > RHO_1_MAX:
    #      rho_1 = RHO_1_MAX
    return rho_1


def logLikelihood(z, k, theta):
    """
    Compute logLikelihood of the pseudo-data
    """
    try:
        h0_x = h_function(z=z,
                          m=z.shape[1],
                          mu=0,
                          sigma_sq=1,
                          rho=0)
        h1_x = h_function(z=z,
                          m=z.shape[1],
                          mu=theta['mu_1'],
                          sigma_sq=theta['sigma_1'],
                          rho=theta['rho_1'])
        ll = 0.0
        for i in range(z.shape[0]):
            ll += (1.0-float(k[i])) * (math.log(1-theta['pi_1']) +
                                       math.log(h0_x[i]))
            ll += float(k[i]) * (math.log(theta['pi_1']) +
                                 math.log(h1_x[i]))
        return ll
    except ValueError as e:
        print("error: logLikelihood: " + str(e))
        print(h1_x[i])
        print(theta)
        quit(-1)
    except Exception as e:
        print("error: logLikelihood: " + str(e))
        print(h1_x[i])
        print(theta)
        quit(-1)


def EM_pseudo_data(z,
                   log,
                   theta,
                   k,
                   threshold=0.001):
    """
    EM optimization of theta for pseudo-data
    """
    theta_t1 = deepcopy(theta)
    ll = 0.0
    ll_t1 = -np.inf
    while abs(ll - ll_t1) > threshold:
        ll = ll_t1
        theta = deepcopy(theta_t1)
        k = E_step_K(z=z, k=k, theta=theta)
        theta_t1['pi_1'] = M_step_pi_1(k=k)
        theta_t1['mu_1'] = M_step_mu_1(z=z, k=k)
        theta_t1['sigma_1'] = M_step_sigma_1(z=z, k=k, theta=theta_t1)
        theta_t1['rho_1'] = M_step_rho_1(z=z, k=k, theta=theta_t1)
        ll_t1 = logLikelihood(z=z, k=k, theta=theta_t1)
        log['ll'].append(ll_t1)
        log['pi'].append(theta['pi_1'])
        log['mu'].append(theta['mu_1'])
        log['sigma'].append(theta['sigma_1'])
        log['rho'].append(theta['rho_1'])
        if ll_t1 - ll < 0.0:
            print("warning: EM decreassing logLikelihood: " + str(ll_t1 - ll))
            print(theta_t1)
            #  theta_t1 = THETA_INIT
    return (theta_t1, k, log)

def plot_log(log, file_name):
    """
    plot logs into a file
    """
    pylab.subplot(5, 1, 1)
    pylab.plot(np.linspace(start=0, stop=len(log['ll']), num=len(log['ll'])), log['ll'])
    pylab.ylabel('logLikelihood')
    pylab.subplot(5, 1, 2)
    pylab.plot(np.linspace(start=0, stop=len(log['ll']), num=len(log['ll'])), log['pi'])
    pylab.ylabel('pi_1')
    pylab.subplot(5, 1, 3)
    pylab.plot(np.linspace(start=0, stop=len(log['ll']), num=len(log['ll'])), log['mu'])
    pylab.ylabel('mu_1')
    pylab.subplot(5, 1, 4)
    pylab.plot(np.linspace(start=0, stop=len(log['ll']), num=len(log['ll'])), log['sigma'])
    pylab.ylabel('sigma_1')
    pylab.subplot(5, 1, 5)
    pylab.plot(np.linspace(start=0, stop=len(log['ll']), num=len(log['ll'])), log['rho'])
    pylab.ylabel('rho_1')
    pylab.savefig(file_name)

def plot_classif(x, u, z, lidr, file_name):
    """
    plot logs into a file
    """
    pylab.subplot(4, 1, 1)
    pylab.hist(x[:, 0], bins=1000, label=str(0))
    pylab.subplot(4, 1, 2)
    pylab.hist(z[:, 0], bins=1000, label=str(0))
    pylab.subplot(4, 1, 3)
    pylab.scatter(x[:, 1], z[:, 0], c=lidr)
    pylab.subplot(4, 1, 4)
    pylab.scatter(u[:, 1], z[:, 0], c=lidr)
    pylab.savefig(file_name)

def pseudo_likelihood(x, threshold=0.001):
    """
    pseudo likelhood optimization for the copula model parameters
    """
    theta = THETA_INIT
    k = [0.0] * int(x.shape[0])
    u = [0.0] * int(x.shape[0])
    z = [0.0] * int(x.shape[0])
    lidr = [0.0] * int(x.shape[0])
    u = compute_empirical_marginal_cdf(compute_rank(x))
    log = {'ll': list(),
           'pi': list(),
           'mu': list(),
           'sigma': list(),
           'rho': list()}
    ll = 0.0
    ll_t1 = -np.inf
    try:
        while abs(ll - ll_t1) > threshold:
            ll = ll_t1
            z = compute_z_from_u(u=u, k=k, theta=theta)
            (theta, k, loglik) = EM_pseudo_data(z=z,
                                                log=log,
                                                k=k,
                                                theta=theta,
                                                threshold=threshold)
            lidr = local_idr(z=z, lidr=lidr, theta=theta)
            ll_t1 = logLikelihood(z=z, k=k, theta=theta)
            log['ll'].append(ll_t1)
            log['pi'].append(theta['pi_1'])
            log['mu'].append(theta['mu_1'])
            log['sigma'].append(theta['sigma_1'])
            log['rho'].append(theta['rho_1'])
            if ll_t1 - ll < 0.0:
                print("warning: pseudo data decreassing logLikelihood: " + str(ll_t1 - ll))
                print(theta)
            plot_log(log, "log_paper.pdf")
            plot_classif(x, u, z, lidr, "classif_paper.pdf")
    except Exception as e:
        plot_log(log, "log_paper.png")
        plot_classif(x, u, z, lidr, "classif_paper.pdf")
    plot_classif(x, u, z, lidr, "classif_paper.pdf")
    return theta
    #  return (theta, lidr)

theta_test = {'pi_1': 0.5, 'mu_1': 5.0, 'sigma_1': 1.0, 'rho_1': 0.9}

#  print(EM_pseudo_data(sim_m_samples(n=10000,
#                                     m=5,
#                                     mu=theta_test['mu_1'],
#                                     sigma_sq=theta_test['sigma_1'],
#                                     rho=theta_test['rho_1'],
#                                     pi=theta_test['pi_1'])["X"],
#                       k=[0.0] * 10000,
#                       theta=THETA_INIT,
#                       threshold=0.001)[0])
print(pseudo_likelihood(sim_m_samples(n=10000,
                                      m=2,
                                      mu=theta_test['mu_1'],
                                      sigma_sq=theta_test['sigma_1'],
                                      rho=theta_test['rho_1'],
                                      pi=theta_test['pi_1'])["X"],
                        threshold=0.01))
print(theta_test)

#  x = sim_m_samples(n=10,
#                                     m=5,
#                                     mu=theta_test['mu_1'],
#                                     sigma_sq=theta_test['sigma_1'],
#                                     rho=theta_test['rho_1'],
#                                     pi=theta_test['pi_1'])["X"]
#  print(x)
#  print(compute_rank(x))
#  print(compute_empirical_marginal_cdf(compute_rank(x)))
