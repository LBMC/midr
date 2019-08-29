#!/usr/bin/python3

"""
Package to compute IDR from n replicates
"""

import math
from copy import deepcopy
from os import path
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
import numpy as np
import pandas as pd
from pynverse import inversefunc
import matplotlib.pyplot as plt

THETA_INIT = {'pi': 0.5,
              'mu': 0.0,
              'sigma': 1.0,
              'rho': 0.9}


def add_log(log, theta, logl, pseudo):
    """
    function to append thata and ll value to the logs
    """
    log['logl'].append(logl)
    if pseudo:
        log['pseudo_data'].append('#FF4949')
    else:
        log['pseudo_data'].append('#4970FF')
    for parameters in theta:
        log[parameters].append(theta[parameters])
    return log


def plot_log(log, file_name):
    """
    plot logs into a file
    """
    x_axis = np.linspace(start=0,
                         stop=len(log['logl']),
                         num=len(log['logl']))
    i = 1
    for parameters in log:
        if parameters != "pseudo_data":
            plt.subplot(len(log.keys()), 1, i)
            plt.scatter(x_axis,
                        log[parameters],
                        c=log['pseudo_data'],
                        s=2)
            plt.ylabel(parameters)
            plt.xlabel('steps')
            i += 1
    plt.savefig(file_name)


def plot_classif(x_score, u_values, z_values, lidr, file_name):
    """
    plot logs into a file
    """
    plt.subplot(4, 1, 1)
    plt.hist(x_score[:, 0], bins=1000, label=str(0))
    plt.subplot(4, 1, 2)
    plt.hist(z_values[:, 0], bins=1000, label=str(0))
    plt.subplot(4, 1, 3)
    plt.scatter(x_score[:, 1], z_values[:, 0], c=lidr)
    plt.subplot(4, 1, 4)
    plt.scatter(u_values[:, 1], z_values[:, 0], c=lidr)
    plt.savefig(file_name)

class NarrowPeaks:
    """
    Class to handle narrowpeak file
    """

    colnum_names = ['chr', 'start', 'stop', 'name', 'score', 'strand',
                    'signalValue', 'pValue', 'qValue', 'peak']

    def __init__(self, file_merge, file_names, score='signalValue'):
        """
        Create narrowpeak DataFrame
        """
        self.files = {'coords': pd.DataFrame(columns=self.colnum_names)}
        self.files_merged = dict()
        if score in self.colnum_names[6:9]:
            self.score = score
        else:
            print("error: " + str(score) +
                  " must be a narrowpeak score column")
            quit(-1)
        self.file_merge, self.file_merge_path = \
            path.normpath(str(file_merge)).path.split(path.sep)
        self.file_names = dict()
        for full_path in file_names:
            file_name, file_path = \
                path.normpath(str(full_path)).path.split(path.sep)
            self.file_names[file_name] = file_path
        self.read_peaks()
        self.sort_peaks()
        self.merge_peaks()
        self.idr()

    def read_line(self, line, file_name):
        """
        read coords from narrowpeak
        """
        line = line.split()
        file_line = dict()
        i = 0
        for col in self.colnum_names:
            file_line[col] = line[i]
            i += 1
        self.files[file_name].append(file_line)

    def read_peaks(self):
        """
        read peak file
        """
        file_merge = str(self.file_merge_path) + str(self.file_merge)
        with open(file_merge, 'r') as fmerge:
            for line in fmerge:
                self.read_line(line, 'coords')
        for file_name in self.file_names:
            file_path = str(self.file_names[file_name]) + str(file_name)
            self.files[file_name] = pd.DataFrame(columns=self.colnum_names)
            with open(file_path, 'r') as ffile:
                for line in ffile:
                    self.read_line(line, file_name)

    def sort_peaks(self):
        """
        sort peaks by chr, start, stop, strand and peaks
        """
        for file_name in self.files:
            self.files[file_name] = self.files[file_name].\
                sort_values(by=np.array(self.colnum_names)[[0, 1, 2, 3, 5, 9]])

    def is_match(self, index_ref, index_file, file_name):
        """
        helper function for merge peaks
        return True if the peak is within the merged peak file
        """
        ref_strand = self.files['coords']['strand'][index_ref]
        peak = self.files[file_name]['peak'][index_file]
        if peak != ref_strand:
            return False
        start = self.files['coords']['start'][index_ref]
        stop = self.files['coords']['stop'][index_ref]
        peak_strand = self.files[file_name]['strand'][index_file]
        if peak < start or peak > stop:
            return False
        return True

    def merge_line(self, index_ref, index_file, file_name):
        """
        merge current file line with merged peak file
        """
        line = self.files['coords'][index_ref]
        for col in np.array(self.colnum_names)[[3, 4, 6, 7, 8, 9]]:
            line[col] = self.files[file_name][index_file][col]
        return line

    def merge_peaks(self):
        """
        merge peaks according to the merged files
        """
        for index_ref, row_ref in self.files['coords'].iterrows():
            for file_name in self.file_names:
                merged_name = str(file_name) + "_merged"
                self.files_merged[merged_name] = \
                    pd.DataFrame(columns=self.colnum_names)
                for index_file, row_file in self.files[file_name].iterrows():
                    if self.is_match(index_ref, index_file, file_name):
                        self.files_merged[merged_name].append(
                            self.merge_line(index_ref, index_file, file_name)
                        )

    def idr(self):
        """
        compute IDR for given score
        """
        data = np.array()
        for file_name in self.files_merged:
            if data.shape[1] == 0:
                data = np.array(self.files_merged[file_name][self.score])
            else:
                data[:, :-1] = np.array(self.files_merged[file_name][[self.score]])
        theta, lidr, k = pseudo_likelihood(x_score=data,
                                             threshold=0.01,
                                             log_name=self.file_merge)
        print(theta)
        i = 0
        for file_name in self.files_merged:
            self.files_merged[file_name]['idr'] = lidr[:, i]
            i += 1

    def write_file(self):
        """
        write output
        """
        for file_name in self.files_merged:
            output_name = str(self.files_merged[file_name]) + \
                str( path.sep ) + "idr" + str(file_name)
            self.files_merged[file_name].to_csv(output_name,
                                                sep='\t',
                                                encoding='utf-8')


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


def sim_m_samples(n_value, m_sample, theta_0, theta_1):
    """
    simulate sample where position score are drawn from two different
    multivariate Gaussian distribution
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
    #  def g_m1(u):
    #      g_func = lambda x: g_function(x, theta=theta) - u
    #      if u == 0.0:
    #          u += np.finfo(float).eps * 4
    #      if u == 1.0:
    #          u -= np.finfo(float).eps * 4
    #      try:
    #          return brentq(f=g_func,
    #                        a=min([-4, theta['mu'] - 4]),
    #                        b=max([4, theta['mu'] + 4]),
    #                        maxiter=1000,
    #                        xtol=np.finfo(float).eps * 4,
    #                        rtol=np.finfo(float).eps * 4)
    #      except ValueError as err:
    #          print("error: compute_z_from_u: " + str(err))
    #          print(theta)
    #          print({'u': u,
    #                 'f(a)': g_func(min([-4, theta['mu'] - 4]))-u,
    #                 'f(b)': g_func(max([4, theta['mu'] + 4]))-u})
    #          quit(-1)
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
        print("error: logLikelihood: " + str(err))
        print(h1_x[i])
        print(theta)
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
                   log,
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
            print("warning: EM decreassing logLikelihood rho: " +
                  str(logl_t1 - logl_t0))
            print(theta_t1)
            return (theta_t0, k_state, log)
        log = add_log(log=log,
                      theta=theta_t1,
                      logl=logl_t1,
                      pseudo=False)
    return (theta_t1, k_state, log)


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
    log = {'logl': list(),
           'pi': list(),
           'mu': list(),
           'sigma': list(),
           'rho': list(),
           'pseudo_data': list()}
    logl_t0 = 0.0
    logl_t1 = -np.inf
    u_values = compute_empirical_marginal_cdf(compute_rank(x_score))
    while delta(theta_t0, theta_t1, threshold, logl_t1):
        logl_t0 = logl_t1
        del theta_t0
        theta_t0 = deepcopy(theta_t1)
        z_values = compute_z_from_u(u_values=u_values,
                                    theta=theta_t1)
        (theta_t1, k_state, log) = em_pseudo_data(z_values=z_values,
                                                  log=log,
                                                  k_state=k_state,
                                                  theta=theta_t1,
                                                  threshold=threshold)
        lidr = local_idr(z_values=z_values,
                         lidr=lidr,
                         theta=theta_t1)
        logl_t1 = loglikelihood(z_values=z_values,
                                k_state=k_state,
                                theta=theta_t1)
        log = add_log(log=log,
                      theta=theta_t1,
                      logl=logl_t1,
                      pseudo=True)
        plot_log(log, "log_" + log_name + ".pdf")
        plot_classif(x_score,
                     u_values,
                     z_values,
                     lidr,
                     "classif_" + log_name + ".pdf")
    print(theta_t1)
    return (theta_t1, lidr, k_state)


THETA_TEST_0 = {'pi': 0.6, 'mu': 0.0, 'sigma': 1.0, 'rho': 0.0}
THETA_TEST_1 = {'pi': 0.6, 'mu': 4.0, 'sigma': 3.0, 'rho': 0.75}
THETA_TEST = {'pi': 0.2,
              'mu': THETA_TEST_1['mu'] - THETA_TEST_0['mu'],
              'sigma': THETA_TEST_0['sigma'] / THETA_TEST_1['sigma'],
              'rho': 0.75}

DATA = sim_m_samples(n_value=10000,
                     m_sample=2,
                     theta_0=THETA_TEST_0,
                     theta_1=THETA_TEST_1)
(THETA_RES, LIDR, K) = pseudo_likelihood(DATA["X"],
                                         threshold=0.01,
                                         log_name=str(THETA_TEST))
print(THETA_TEST)

plt.subplot(1, 1, 1)
plt.scatter(DATA['K'], K, c=LIDR)
plt.ylabel('k')
plt.savefig("k_vs_estK_" + str(THETA_TEST) + ".pdf")
