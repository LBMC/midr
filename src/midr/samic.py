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

from sys import float_info
from scipy.optimize import minimize, Bounds
import numpy as np
import midr.log as log
import midr.archimedean as archimedean
from midr.idr import compute_empirical_marginal_cdf, compute_rank, sim_m_samples


def delta(copula_list, params_list, threshold):
    """
    Return true if the difference between two iteration of samic if less than
    the threhsold
    :param copula_list: str with the copula name
    :param params_list: list of model parameters
    :param threshold: flood withe the minimal difference to reach
    :return: bool
    """
    max_delta = list()
    max_delta.append(max(abs(params_list['alpha'] - params_list['alpha_old'])))
    for copula in copula_list:
        max_delta.append(max([
            abs(params_list[copula]['pi'] - params_list[copula]['pi_old']),
            abs(params_list[copula]['theta'] - params_list[copula]['theta_old'])
        ]))
    return max(max_delta) >= threshold


def expectation_k(u_values, copula, params_list):
    """
    compute proba for each line to be in one component or the other
    :param u_values:
    :param copula:
    :param params_list:
    :return:
    """
    copula_density = {
        'clayton': archimedean.pdf_clayton,
        'frank': archimedean.pdf_frank,
        'gumbel': archimedean.pdf_gumbel
    }
    k_state = params_list[copula]['pi'] / (
        params_list[copula]['pi'] +
        (1.0 - params_list[copula]['pi']) * copula_density[copula](
            u_values,
            params_list[copula]['theta'],
        )
    )
    return np.minimum(k_state, 1.0)


def expectation_l(u_values, copula_list, params_list):
    """
    compute proba for each copula mix to describe the data
    :param u_values:
    :param copula_list:
    :param params_list:
    :return:
    """
    copula_density = {
        'clayton': archimedean.pdf_clayton,
        'frank': archimedean.pdf_frank,
        'gumbel': archimedean.pdf_gumbel
    }
    l_state = np.zeros((u_values.shape[0], len(copula_list)))
    dcopula = np.zeros((u_values.shape[0], len(copula_list)))
    for i in range(len(copula_list)):
        dcopula[:, i] = (params_list['alpha'][i] *
                         params_list[copula_list[i]]['pi'] *
                         copula_density[copula_list[i]](
                             u_values,
                             params_list[copula_list[i]]['theta'],
                         )
                         )
    for i in range(len(copula_list)):
        l_state[:, i] = dcopula[:, i] / np.sum(dcopula, axis=1)
    return l_state


def density_mix(theta, u_values, copula):
    """
    pdf of the samic mixture for a given copula
    :param u_values:
    :param copula:
    :param theta:
    :return:
    """
    copula_density = {
        'clayton': archimedean.pdf_clayton,
        'frank': archimedean.pdf_frank,
        'gumbel': archimedean.pdf_gumbel
    }
    return -np.sum(
        copula_density[copula](
            u_values,
            theta,
            is_log=True
        ),
        axis=0
    )


def loglikelihood(u_values, copula_list, params_list):
    """
    compute loklik for samic model
    :param u_values:
    :param copula_list:
    :param params_list:
    :return:
    """
    copula_density = {
        'clayton': archimedean.pdf_clayton,
        'frank': archimedean.pdf_frank,
        'gumbel': archimedean.pdf_gumbel
    }
    loglik = 0.0
    for copula in copula_list:
        loglik = -np.sum(
            np.log(params_list[copula]['pi']) +
            np.log(params_list['alpha'][params_list['order'][copula]]) +
            copula_density[copula](
                u_values,
                params_list[copula]['theta'],
                is_log=True
            ),
            axis=0
        )
    return loglik


def local_idr(u_values, copula_list, params_list):
    """
    Compute local idr for the samic method
    :param u_values:
    :param copula_list:
    :param params_list:
    :return:
    """
    copula_density = {
        'clayton': archimedean.pdf_clayton,
        'frank': archimedean.pdf_frank,
        'gumbel': archimedean.pdf_gumbel
    }
    lidr = np.zeros(u_values.shape[0])
    lidr_denum = np.zeros(u_values.shape[0])
    for i in range(len(copula_list)):
        lidr += (params_list['alpha'][i] *
                 (1.0 - params_list[copula_list[i]]['pi']))
        lidr_denum += (params_list['alpha'][i] *
                       params_list[copula_list[i]]['pi'] *
                       copula_density[copula_list[i]](
                           u_values,
                           params_list[copula_list[i]]['theta'],
                       )
                       )
    return lidr / (lidr + lidr_denum)


def minimize_pi(k_state):
    """
    compute maximization of pi
    """
    return float(sum(k_state)) / float(len(k_state))


def minimize_alpha(l_state):
    """
    compute maximization of pi
    """
    alpha = np.zeros(l_state.shape[1])
    alpha[:-1] = np.sum(l_state[:, :-1], axis=0) / float(l_state.shape[0])
    alpha[-1] = 1.0 - sum(alpha)
    return alpha


def constraint(x, theta_min=np.nan, theta_max=np.nan, return_min=True,
               eps=float_info.min):
    """
    compute contraint for theta ineq
    :param x:
    :param theta_min:
    :param theta_max:
    :param return_min
    :param eps:
    :return:
    """
    interval = [x - (theta_min + eps), (theta_max - eps) - x]
    if interval[0] > interval[1]:
        interval[1] = interval[0] + 1.0
    if return_min:
        return interval[0]
    else:
        return interval[1]


def build_constraint(copula, old_theta=np.nan, eps=1.0):
    """
    write consts for a given copula
    :param copula:
    :param old_theta:
    :param eps
    :return:
    """
    thetas = {
        'clayton': {
            'theta_min': 0.0,
            'theta_max': 1000.0
        },
        'frank': {
            'theta_min': 0.0,
            'theta_max': 745.0
        },
        'gumbel': {
            'theta_min': 1.0,
            'theta_max': 100.0
        }
    }

    def consts_min(x):
        """
        Return minimum for the range
        :param x:
        :return:
        """
        return constraint(
            x=x,
            theta_min=thetas[copula]['theta_min'],
            theta_max=thetas[copula]['theta_max'],
        )

    def consts_max(x):
        """
        Return maximum for the range
        :param x:
        :return:
        """
        return constraint(
            x=x,
            theta_min=thetas[copula]['theta_min'],
            theta_max=thetas[copula]['theta_max'],
            return_min=False
        )

    return [
        {'type': 'ineq', 'fun': consts_min},
        {'type': 'ineq', 'fun': consts_max}
    ]

def build_bounds(copula, eps=1e-8):
    """
    return set of bound for a given copula
    :param copula:
    :param eps:
    :return:
    """
    thetas = {
        'clayton': {
            'theta_min': 0.0,
            'theta_max': 1000.0
        },
        'frank': {
            'theta_min': 0.0,
            'theta_max': 745.0
        },
        'gumbel': {
            'theta_min': 1.0,
            'theta_max': 100
        }
    }
    return [
        (thetas[copula]['theta_min'] + eps, thetas[copula]['theta_max'] - eps)
    ]


def minimize_theta(u_values, copula, params_list):
    """
    find theta that minimize the likelihood of the copula density
    :param u_values:
    :param copula:
    :param params_list:
    :return:
    """
    old_theta = params_list[copula]['theta']
    res = minimize(
        fun=density_mix,
        args=(u_values, copula),
        x0=old_theta,
        constraints=build_constraint(copula)
    )
    if np.isnan(res.x):
        return old_theta
    else:
        return res.x[0]


def samic(x_score, threshold=1e-4, log_name=""):
    """
    implementation of the samic method for m samples
    :param x_score np.array of score (measures x samples)
    :param threshold float min delta between every parameters between two
    iterations
    :param log_name:
    :return (theta: dict, lidr: list) with theta the model parameters and
    lidr the local idr values for each measures
    >>> THETA_TEST_0 = {'mu': 0.0, 'sigma': 1.0, 'rho': 0.0}
    >>> THETA_TEST_1 = {'pi': 0.1, 'mu': 4.0, 'sigma': 3.0, 'rho': 0.75}
    >>> DATA = sim_m_samples(n_value=10000,
    ...                      m_sample=4,
    ...                      theta_0=THETA_TEST_0,
    ...                      theta_1=THETA_TEST_1)
    >>> lidr = samic(DATA["X"], threshold=0.0001)
    >>> np.sum((lidr < 0.5).all() == DATA["K"]) / len(lidr)
    """
    u_values = compute_empirical_marginal_cdf(compute_rank(x_score))
    copula_list = ["clayton", "frank", "gumbel"]
    dmle_copula = {
        'clayton': archimedean.dmle_copula_clayton,
        'frank': lambda x: 5,  # since the dmle for frank is slow...
        'gumbel': archimedean.dmle_copula_gumbel
    }
    params_list = dict()
    params_list['order'] = dict()
    i = 0
    copula = copula_list[0]
    for copula in copula_list:
        params_list[copula] = {
            'theta': dmle_copula[copula](u_values),
            'theta_old': np.nan,
            'pi': 0.5,
            'pi_old': np.Inf,
            'k_state': np.zeros(u_values.shape[0])
        }
        params_list['order'][copula] = i
        i += 1
    params_list['l_state'] = np.zeros((u_values.shape[0], len(copula_list)))
    params_list['alpha'] = np.repeat(1.0 / len(copula_list), len(copula_list))
    params_list['alpha_old'] = np.repeat(np.Inf, len(copula_list))
    while delta([copula], params_list, threshold):
        params_list['alpha_old'] = params_list['alpha']
        for copula in copula_list:
            params_list[copula]['pi_old'] = params_list[copula]['pi']
            params_list[copula]['theta_old'] = params_list[copula]['theta']
            params_list[copula]['k_state'] = expectation_k(
                u_values=u_values,
                copula=copula,
                params_list=params_list,
            )
            params_list[copula]['pi'] = minimize_pi(
                k_state=params_list[copula]['k_state']
            )
            params_list[copula]['theta'] = minimize_theta(
                u_values=u_values,
                copula=copula,
                params_list=params_list
            )
        params_list['l_state'] = expectation_l(
            u_values=u_values,
            copula_list=copula_list,
            params_list=params_list,
        )
        params_list['alpha'] = minimize_alpha(
            l_state=params_list['l_state']
        )
        log.logging.info("%s", log_samic(params_list, copula_list))
    return local_idr(
        u_values=u_values,
        copula_list=copula_list,
        params_list=params_list
    )


def log_samic(params_list, copula_list):
    """
    return str of pseudo_likelihood parameter estimate
    :param params_list:
    :param copula_list:
    :return:
    """
    log_str = str('{' +
                  '"alpha": ' + str(params_list['alpha'])
                  )
    for copula in copula_list:
        log_str += str(', "' + copula + '": {'
                                      '"theta": ' + str(
            params_list[copula]['theta']) + ', ' +
                       '"pi": ' + str(params_list[copula]['pi']) + ', ' +
                       '}')
    return log_str + '}'


if __name__ == "__main__":
    import doctest

    doctest.testmod()
