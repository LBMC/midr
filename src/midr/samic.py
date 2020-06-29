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
from scipy.optimize import minimize
import numpy as np
import midr.log as log
import midr.archimedean as archimedean
from midr.idr import compute_empirical_marginal_cdf, compute_rank, sim_m_samples
import midr.archimediean_plots as archimediean_plots

COPULA_DENSITY = {
    'clayton': archimedean.pdf_clayton,
    'frank': archimedean.pdf_frank,
    'gumbel': archimedean.pdf_gumbel
}
DMLE_COPULA = {
    'clayton': archimedean.dmle_copula_clayton,
    'frank': archimedean.dmle_copula_frank,
    'gumbel': archimedean.dmle_copula_gumbel
}


def delta(params_list, threshold):
    """
    Return true if the difference between two iteration of samic if less than
    the threhsold
    :param params_list: list of model parameters
    :param threshold: flood withe the minimal difference to reach
    :return: bool
    """
    max_delta = list()
    max_delta.append(max(abs(params_list['alpha'] - params_list['alpha_old'])))
    for copula in COPULA_DENSITY.keys():
        max_delta.append(max([
            abs(params_list[copula]['pi'] - params_list[copula]['pi_old']),
            abs(params_list[copula]['theta'] - params_list[copula]['theta_old'])
        ]))
    return max(max_delta) >= threshold


def expectation_l(u_values, params_list):
    """
    compute proba for each copula mix to describe the data
    :param u_values:
    :param params_list:
    :return:
    """
    l_state = np.zeros((u_values.shape[0], len(COPULA_DENSITY)))
    dcopula = np.zeros((u_values.shape[0], len(COPULA_DENSITY)))
    for copula in COPULA_DENSITY.keys():
        dcopula[:, params_list['order'][copula]] = (
                params_list['alpha'][params_list['order'][copula]] *
                (
                        params_list[copula]['pi'] +
                        (1 - params_list[copula]['pi']) *
                        COPULA_DENSITY[copula](
                            u_values,
                            params_list[copula]['theta'],
                        )
                )
        )
    for copula in COPULA_DENSITY.keys():
        l_state[:, params_list['order'][copula]] = \
            dcopula[:, params_list['order'][copula]] / np.sum(dcopula, axis=1)
    return l_state


def density_pi(pi, u_values, copula, params_list):
    """
    pdf of the samic mixture for a given copula
    :param u_values:
    :param copula:
    :param theta:
    :return:
    >>> density_pi(
    ...    pi=0.2,
    ...    u_values = np.array([
    ...       [0.42873569, 0.18285458, 0.9514195],
    ...       [0.25148149, 0.05617784, 0.3378213],
    ...       [0.79410993, 0.76175687, 0.0709562],
    ...       [0.02694249, 0.45788802, 0.6299574],
    ...       [0.39522060, 0.02189511, 0.6332237],
    ...       [0.66878367, 0.38075101, 0.5185625],
    ...       [0.90365653, 0.19654621, 0.6809525],
    ...       [0.28607729, 0.82713755, 0.7686878],
    ...       [0.22437343, 0.16907646, 0.5740400],
    ...       [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    copula = "frank",
    ...    params_list = {"frank": {'theta': 2}}
    ... )
    2.0863120778575763
    """
    return -np.sum(
        np.log(
            pi +
            (1 - pi) *
            COPULA_DENSITY[copula](
                u_values=u_values,
                theta=params_list[copula]['theta']
            )
        ),
        axis=0
    )


def density_theta(theta, u_values, copula, params_list):
    """
    pdf of the samic mixture for a given copula
    :param u_values:
    :param copula:
    :param theta:
    :return:
    >>> density_theta(
    ...    theta=2,
    ...    u_values = np.array([
    ...       [0.42873569, 0.18285458, 0.9514195],
    ...       [0.25148149, 0.05617784, 0.3378213],
    ...       [0.79410993, 0.76175687, 0.0709562],
    ...       [0.02694249, 0.45788802, 0.6299574],
    ...       [0.39522060, 0.02189511, 0.6332237],
    ...       [0.66878367, 0.38075101, 0.5185625],
    ...       [0.90365653, 0.19654621, 0.6809525],
    ...       [0.28607729, 0.82713755, 0.7686878],
    ...       [0.22437343, 0.16907646, 0.5740400],
    ...       [0.66752741, 0.69487362, 0.3329266]
    ...    ]),
    ...    copula = "frank",
    ...    params_list = {"frank": {'pi': 0.16}}
    ... )
    2.238971254704331
    """
    return -np.sum(
        np.log(
            params_list[copula]['pi'] +
            (1 - params_list[copula]['pi']) *
            COPULA_DENSITY[copula](
                u_values=u_values,
                theta=theta
            )
        ),
        axis=0
    )


def local_idr(u_values, params_list):
    """
    Compute local idr for the samic method
    :param u_values:
    :param params_list:
    :return:
    """
    lidr = 0.0
    lidr_denum = 0.0
    for copula in COPULA_DENSITY.keys():
        lidr += params_list['alpha'][params_list['order'][copula]] * \
                params_list[copula]['pi']
        lidr_denum += params_list['alpha'][params_list['order'][copula]] * \
                      (1 - params_list[copula]['pi']) * \
                      COPULA_DENSITY[copula](
                          u_values,
                          params_list[copula]['theta'],
                      )
    return lidr / (lidr + lidr_denum)


def minimize_alpha(l_state):
    """
    compute maximization of alpha
    """
    alpha = np.zeros(l_state.shape[1])
    alpha[:-1] = np.sum(l_state[:, :-1], axis=0) / float(l_state.shape[0])
    alpha[-1] = 1.0 - sum(alpha)
    return alpha


def build_bounds(copula, eps=1e-4):
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
    return (
        thetas[copula]['theta_min'] + eps, thetas[copula]['theta_max'] - eps
    )


def minimize_pi(u_values, copula, params_list):
    """
    find theta that minimize the likelihood of the copula density
    :param u_values:
    :param copula:
    :param params_list:
    :return:
    """
    log.logging.debug("%s", copula + " minimize_pi")
    old_pi = params_list[copula]['pi']
    log.logging.debug("%s", str(old_pi) + " old_pi")
    if log.logging.root.level == log.logging.DEBUG:
        archimediean_plots.pdf_copula_plot(
            lower=0.0,
            upper=1.0,
            copula=copula,
            pdf_function=density_pi,
            u_values=u_values,
            params_list=params_list,
        )
    res = minimize(
        fun=density_pi,
        args=(u_values, copula, params_list),
        x0=old_pi,
        bounds=[(0.0, 1.0)],
    )
    log.logging.debug("%s", res)
    if np.isnan(res.x[0]):
        log.logging.debug("%s", str(old_pi) + " new_pi = old_pi")
        return old_pi
    else:
        log.logging.debug("%s", str(res.x[0]) + " new_pi")
        return res.x[0]


def minimize_theta(u_values, copula, params_list):
    """
    find theta that minimize the likelihood of the copula density
    :param u_values:
    :param copula:
    :param params_list:
    :return:
    """
    log.logging.debug("%s", copula + " minimize_theta")
    old_theta = params_list[copula]['theta']
    log.logging.debug("%s", str([build_bounds(copula)]) + " minimize() bounds")
    log.logging.debug("%s", str(old_theta) + " old_theta")
    if log.logging.root.level == log.logging.DEBUG:
        archimediean_plots.pdf_copula_plot(
            lower=build_bounds(copula)[0],
            upper=build_bounds(copula)[1],
            copula=copula,
            pdf_function=density_theta,
            u_values=u_values,
            params_list=params_list,
        )
    res = minimize(
        fun=density_theta,
        args=(u_values, copula, params_list),
        x0=old_theta,
        bounds=[build_bounds(copula)],
    )
    log.logging.debug("%s", res)
    if np.isnan(res.x[0]):
        log.logging.debug("%s", str(old_theta) + " new_theta = old_theta")
        return old_theta
    else:
        log.logging.debug("%s", str(res.x[0]) + " new_theta")
        return res.x[0]


def samic(x_score, threshold=1e-4):
    """
    implementation of the samic method for m samples
    :param x_score np.array of score (measures x samples)
    :param threshold float min delta between every parameters between two
    iterations
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
    log.logging.info("%s", "computing idr")
    u_values = compute_empirical_marginal_cdf(compute_rank(x_score))
    params_list = dict()
    params_list['order'] = dict()
    i = 0
    for copula in COPULA_DENSITY.keys():
        params_list[copula] = {
            'theta': DMLE_COPULA[copula](u_values),
            'theta_old': np.nan,
            'pi': 0.5,
            'pi_old': np.Inf,
            'k_state': np.zeros(u_values.shape[0])
        }
        params_list['order'][copula] = i
        i += 1
    params_list['l_state'] = np.zeros((u_values.shape[0], len(COPULA_DENSITY)))
    params_list['alpha'] = np.repeat(1.0 / len(COPULA_DENSITY), len(COPULA_DENSITY))
    params_list['alpha_old'] = params_list['alpha'][:]
    while delta(params_list, threshold):
        for copula in COPULA_DENSITY:
            params_list[copula]['pi_old'] = params_list[copula]['pi']
            params_list[copula]['theta_old'] = params_list[copula]['theta']
            params_list[copula]['pi'] = minimize_pi(
                u_values=u_values,
                copula=copula,
                params_list=params_list
            )
            params_list[copula]['theta'] = minimize_theta(
                u_values=u_values,
                copula=copula,
                params_list=params_list
            )
            log.logging.info(
                "%s %s",
                copula,
                log_samic(params_list)
            )
        params_list['alpha_old'] = params_list['alpha'][:]
        params_list['l_state'] = expectation_l(
            u_values=u_values,
            params_list=params_list,
        )
        params_list['alpha'] = minimize_alpha(
            l_state=params_list['l_state']
        )
        log.logging.info(
            "%s",
            log_samic(params_list)
        )
    return local_idr(
        u_values=u_values,
        params_list=params_list
    )


def log_samic(params_list):
    """
    return str of pseudo_likelihood parameter estimate
    :param params_list:
    :return:
    """
    log_str = str('{' +
                  '"alpha": ' + str(params_list['alpha'])
                  )
    for copula in COPULA_DENSITY.keys():
        log_str += str(
            ', \n"' + copula + '": {"theta": ' +
            str(params_list[copula]['theta']) + ', ' +
            '"pi": ' + str(params_list[copula]['pi']) + '}')
    return log_str + ' }'


if __name__ == "__main__":
    import doctest

    doctest.testmod()
