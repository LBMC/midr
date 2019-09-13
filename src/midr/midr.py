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
import sys
from os import path, makedirs, access, R_OK, W_OK
from copy import deepcopy
from pathlib import PurePath
import argparse
import logging
from scipy.stats import rankdata
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pynverse import inversefunc


def setup_logging(options):
    """Configure logging."""
    root = logging.getLogger("")
    root.setLevel(logging.WARNING)
    LOGGER.setLevel(options.debug and logging.DEBUG or logging.INFO)
    if options.verbose:
        message = logging.StreamHandler()
        message.setFormatter(logging.Formatter(
            "%(asctime)s: %(message)s", datefmt='%H:%M:%S'))
        root.addHandler(message)


class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    """
    helper class to make ArgumentParser
    """


def parse_args(args=sys.argv[1:]):
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=CustomFormatter)

    arg = parser.add_argument_group("IDR settings")
    arg.add_argument("--merged", "-m", metavar="FILE",
                     dest='merged',
                     required=True,
                     default=argparse.SUPPRESS,
                     type=str,
                     help="file of the merged NarrowPeaks")
    arg.add_argument("--files", "-f", metavar="FILES",
                     dest='files',
                     required=True,
                     default=argparse.SUPPRESS,
                     type=str,
                     nargs='+',
                     help="list of NarrowPeaks files")
    arg.add_argument("--output", "-o", metavar="DIR",
                     dest='output',
                     required=False,
                     default="results",
                     type=str,
                     help="output directory")
    arg.add_argument("--score", "-s", metavar="SCORE_COLUMN",
                     dest='score',
                     required=False,
                     default='signalValue',
                     type=str,
                     help="NarrowPeaks score column to compute the IDR on, \
                     one of 'score', 'signalValue', 'pValue' or 'qValue'")
    arg.add_argument("--threshold", "-t", metavar="THRESHOLD",
                     dest='threshold',
                     required=False,
                     default=0.01,
                     type=float,
                     help="Threshold value for the precision of the estimator")
    arg.add_argument("--merge_function", "-mf", metavar="MERGE_FUNCTION",
                     dest='merge_function',
                     required=False,
                     default='max',
                     type=str,
                     help="function to determine the score to keep for \
                     overlpping peak ('max', 'mean', 'median', 'min')")
    arg.add_argument("--debug", "-d", action="store_true",
                     default=False,
                     help="enable debugging")
    arg.add_argument("--verbose", "-v", action="store_true",
                     default=False,
                     help="log to console")
    return parser.parse_args(args)


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
    plt.ylabel('counts')
    plt.xlabel('x scores')
    plt.subplot(4, 1, 2)
    plt.hist(z_values[:, 0], bins=1000, label=str(0))
    plt.ylabel('counts')
    plt.xlabel('z scores')
    plt.subplot(4, 1, 3)
    dotplot1 = plt.scatter(x_score[:, 1], z_values[:, 0], c=lidr)
    plt.ylabel('z score')
    plt.xlabel('x scores')
    cbar = plt.colorbar(dotplot1)
    cbar.ax.set_ylabel('lidr')
    plt.subplot(4, 1, 4)
    dotplot2 = plt.scatter(u_values[:, 1], z_values[:, 0], c=lidr)
    plt.ylabel('z score')
    plt.xlabel('u scores')
    cbar = plt.colorbar(dotplot2)
    cbar.ax.set_ylabel('lidr')
    plt.savefig(file_name)


class NarrowPeaks:
    """
    Class to handle narrowpeak file
    """

    column_names = ['chr', 'start', 'stop', 'name', 'score', 'strand',
                    'signalValue', 'pValue', 'qValue', 'peak']
    score_columns = ['score', 'signalValue', 'pValue', 'qValue']
    sort_columns = ['chr', 'strand', 'start', 'stop', 'peak']

    def __init__(self, file_merge, file_names, output, score='signalValue',
                 threshold=0.01,
                 merge_function='mean'):
        """
        Create narrowpeak DataFrame
        """
        self.files = dict()
        self.files_merged = dict()
        if merge_function ==  'mean':
            self.merge_function = np.mean
        elif merge_function == 'median':
            self.merge_function = np.median
        elif merge_function == 'min':
            self.merge_function = min
        else:
            self.merge_function = max
        if score in self.column_names[6:9]:
            self.score = score
        else:
            LOGGER.exception("error: " + str(score) +
                             " must be a NarrowPeak score column " +
                             str(self.score_columns))
            quit(-1)
        file_path = PurePath(file_merge)
        self.file_merge = file_path.name
        self.file_merge_path = file_path.parent
        self.file_names = dict()
        for full_path in file_names:
            file_path = PurePath(full_path)
            assert file_path.name not in self.file_names, \
                "error: file names must be unique (option --file or -f): {}"\
                .format(file_path.name)
            self.file_names[file_path.name] = file_path.parent
        self.output = PurePath(output)
        self.read_peaks()
        self.sort_peaks()
        self.collapse_peaks()
        self.merge_peaks()
        assert access(PurePath(self.output).parent, W_OK), \
            "Folder {} isn't writable".format(self.output)
        if not path.isdir(self.output):
            makedirs(self.output)
        self.idr(threshold=threshold)
        self.write_file()

    def read_peaks(self):
        """
        read peak file
        """
        LOGGER.info("%s", "loading " +
                    str(len(self.file_names) + 1) +
                    " NarrowPeak files...")
        file_path = PurePath(self.file_merge_path).joinpath(self.file_merge)
        assert path.isfile(file_path), \
            "File {} doesn't exist".format(file_path)
        assert access(file_path, R_OK), \
            "File {} isn't readable".format(file_path)
        self.files['coords'] = pd.read_csv(
            file_path,
            sep='\t',
            header=None,
            names=self.column_names
        )
        for file_name in self.file_names:
            file_path = PurePath(self.file_names[file_name])\
                .joinpath(file_name)
            assert path.isfile(file_path), \
                "File {} doesn't exist".format(file_path)
            assert access(file_path, R_OK), \
                "File {} isn't readable".format(file_path)
            self.files[file_name] = pd.read_csv(
                file_path,
                sep='\t',
                header=None,
                names=self.column_names
            )
        LOGGER.info("%s", "loading " +
                    str(len(self.file_names) + 1) +
                    " NarrowPeak done.")

    def sort_peaks(self):
        """
        sort peaks by chr, start, stop, strand and peaks
        """
        LOGGER.info("%s", "sorting " +
                    str(len(self.files)) +
                    " NarrowPeak files...")
        i = 0
        for file_name in self.files:
            self.files[file_name] = self.files[file_name]\
                .sort_values(by=self.sort_columns)
            i += 1
        LOGGER.info("%s", "sorting " +
                    str(len(self.files)) +
                    " NarrowPeak done.")

    def collapse_peaks(self, av_func=max):
        """
        merge overlapping peak in the same file
        """
        for file_name in ['coords'] + list(self.file_names.keys()):
            LOGGER.info("%s", "collapsing " +
                        str(file_name) +
                        " with " +
                        str(self.files[file_name].shape[0]) +
                        " peaks...")
            index_t0 = 0
            index_t1 = 1
            to_merge = list()
            while index_t1 < self.files[file_name].shape[0]:
                if self.peak_overlap(file_name=file_name,
                                     index_ref=index_t0,
                                     index_file=index_t1,
                                     ref=self.files):
                    to_merge.append(index_t0)
                else:
                    if to_merge:
                        to_merge.append(index_t0)
                        to_merge = self.files[file_name].index.intersection(
                            to_merge
                        )
                        self.files[file_name].loc[to_merge[0], self.score] = \
                            av_func(self.files[file_name].loc[to_merge,
                                                              self.score])
                        self.files[file_name] = \
                            self.files[file_name].drop(to_merge[1:])
                        del to_merge
                        to_merge = list()
                index_t0 += 1
                index_t1 += 1
            self.files[file_name] = self.files[file_name].reset_index()
            LOGGER.info("%s", "collapsing " +
                        str(file_name) +
                        " with " +
                        str(self.files[file_name].shape[0]) +
                        " peaks done.")

    def create_empty_merged(self, file_name):
        """
        helper function for merge_peaks
        """
        self.files_merged[file_name] = self.files['coords'].copy()
        self.files_merged[file_name].loc[:, self.score_columns] = -1
        self.files_merged[file_name].drop_duplicates()

    def peak_overlap(self, file_name, index_ref, index_file, ref=None):
        """
        return true if two peak are overlapping and false otherwise
        """
        peak_a = None
        if ref is None:
            peak_a = self.files_merged[file_name].loc[index_ref]
        else:
            peak_a = ref[file_name].loc[index_ref]
        peak_b = self.files[file_name].loc[index_file]
        overlap = False
        if peak_a['strand'] != peak_b['strand']:
            return overlap
        if peak_a['chr'] != peak_b['chr']:
            return overlap
        if (peak_a['start'] <= peak_b['start'] and
                peak_b['start'] <= peak_a['stop']):
            overlap = True
        if (peak_a['start'] <= peak_b['stop'] and
                peak_b['stop'] <= peak_a['stop']):
            overlap = True
        if (peak_b['start'] <= peak_a['start'] and
                peak_a['start'] <= peak_b['stop']):
            overlap = True
        if (peak_b['start'] <= peak_a['stop'] and
                peak_a['stop'] <= peak_b['stop']):
            overlap = True
        return overlap

    def peak_after(self, file_name, index_ref, index_file):
        """
        return True if peak_b is after peak_a and non_overlapping
        """
        peak_a = self.files_merged[file_name].loc[index_ref]
        peak_b = self.files[file_name].loc[index_file]
        if peak_a['chr'] != peak_b['chr']:
            chr_a_index = self.files_merged[file_name]['chr']\
                .unique().tolist().index(peak_a['chr'])
            chr_b_index = self.files[file_name]['chr']\
                .unique().tolist().index(peak_b['chr'])
            if chr_a_index < chr_b_index:
                return True
        if peak_a['strand'] == peak_b['strand']:
            if peak_a['stop'] < peak_b['start']:
                return True
        return False

    def peak_before(self, file_name, index_ref, index_file):
        """
        return True if peak_b is before peak_a and non_overlapping
        """
        peak_b = self.files_merged[file_name].loc[index_ref]
        peak_a = self.files[file_name].loc[index_file]
        if peak_a['chr'] != peak_b['chr']:
            chr_b_index = self.files_merged[file_name]['chr']\
                .unique().tolist().index(peak_b['chr'])
            chr_a_index = self.files[file_name]['chr']\
                .unique().tolist().index(peak_a['chr'])
            if chr_a_index < chr_b_index:
                return True
        if peak_a['strand'] == peak_b['strand']:
            if peak_a['stop'] < peak_b['start']:
                return True
        return False

    def best_peak(self, file_name, index_ref, index_file, index_merge):
        """
        return true if peak_b as a better overlap with peak_a than peak_c
        """
        peak_a = self.files_merged[file_name].loc[index_ref]
        peak_b = self.files[file_name].loc[index_file]
        peak_c = self.files[file_name].loc[index_merge]
        overlap_ab = max([abs(peak_a['stop'] - peak_b['start']),
                          abs(peak_b['stop'] - peak_a['start'])])
        overlap_ac = max([abs(peak_a['stop'] - peak_c['start']),
                          abs(peak_c['stop'] - peak_a['start'])])
        if overlap_ab > overlap_ac:
            return True
        if overlap_ac > overlap_ab:
            return False
        peak_dist_ab = abs((peak_a['start'] + peak_a['peak']) -
                           (peak_b['start'] + peak_b['peak']))
        peak_dist_ac = abs((peak_a['start'] + peak_a['peak']) -
                           (peak_c['start'] + peak_c['peak']))
        if peak_dist_ab > peak_dist_ac:
            return True
        if peak_dist_ac > peak_dist_ab:
            return False
        if peak_b['score'] >= peak_c['score']:
            return True
        return False

    def copy_score(self, file_name, index_ref, index_file):
        """
        function to copy the score column of file_line into ref_line
        """
        for column in self.score_columns:
            self.files_merged[file_name].loc[index_ref, column] = \
                self.files[file_name].loc[index_file, column]

    def merge_overlap(self, file_name, indexes):
        """
        how to merge line in case of overlap
        """
        if self.files_merged[file_name].loc[indexes['ref'],
                                            self.score] != -1:
            if self.best_peak(file_name,
                              indexes['ref'],
                              indexes['file'],
                              indexes['merged']):
                self.copy_score(file_name,
                                indexes['ref'],
                                indexes['file'])
                indexes['merged'] = indexes['file']
        else:
            self.copy_score(file_name,
                            indexes['ref'],
                            indexes['file'])
            indexes['merged'] = indexes['file']
        indexes['file'] += 1
        return indexes

    def merge_non_overlap(self, file_name, indexes):
        """
        how to merge line in case of no overlap
        """
        if self.peak_after(file_name,
                           indexes['ref'],
                           indexes['file']):
            indexes['ref'] += 1
        elif self.peak_before(file_name,
                              indexes['ref'],
                              indexes['file']):
            indexes['file'] += 1
        else:
            LOGGER.exception("%s",
                             "error: merge_non_overlap: merge non overlap")
        return indexes

    def merge_line(self, file_name, indexes):
        """
        merge current file line with merged peak file
        return a dict of the next line to read
        """
        if self.peak_overlap(file_name, indexes['ref'], indexes['file']):
            return self.merge_overlap(file_name,
                                      indexes)
        else:
            return self.merge_non_overlap(file_name,
                                          indexes)

    def drop_line(self):
        """
        remove line with score of -1 in at least one replicate
        """
        rows_to_drop = list()
        for file_name in self.files_merged:
            for index, line in self.files_merged[file_name].iterrows():
                if line[self.score] == -1:
                    rows_to_drop.append(index)
        for file_name in self.files_merged:
            self.files_merged[file_name] = \
                self.files_merged[file_name].drop(rows_to_drop)
            self.files_merged[file_name] = \
                self.files_merged[file_name].reset_index()

    def merge_peaks(self):
        """
        merge peaks according to the merged files
        """
        i = 0
        for file_name in self.file_names:
            LOGGER.info("%s", "building consensus from merged for " +
                        str(file_name) + " " +
                        str(i) + "/" +
                        str(len(self.files) - 1) +
                        " NarrowPeak ...")
            self.create_empty_merged(file_name=file_name)
            indexes = {'ref': 0,
                       'file': 0,
                       'merged': 0}
            while (indexes['ref'] < self.files_merged[file_name].shape[0] and
                   indexes['file'] < self.files[file_name].shape[0]):
                indexes = self.merge_line(
                    file_name=file_name,
                    indexes=indexes,
                )
            i += 1
        peaks_before = self.files_merged[next(iter(self.file_names))].shape[0]
        self.drop_line()
        peaks_after = self.files_merged[next(iter(self.file_names))].shape[0]
        LOGGER.info("%s", "building consensus from merged for " +
                    str(i) + "/" + str(len(self.files) - 1) +
                    " NarrowPeak done. (" +
                    str(peaks_after) + "/" + str(peaks_before) + " peaks)."
                    )

    def idr(self, threshold):
        """
        compute IDR for given score
        """
        data = np.zeros(shape=(self.files_merged[
            next(iter(self.files_merged))].shape[0],
                               len(self.files_merged)))
        LOGGER.info("%s", "computing idr...")
        i = 0
        for file_name in self.files_merged:
            score = np.array(self.files_merged[file_name][self.score])
            data[:, i] = score.astype(float)
            i += 1
        theta, lidr = pseudo_likelihood(x_score=data,
                                        threshold=threshold,
                                        log_name=PurePath(self.output)
                                        .joinpath(self.file_merge))
        LOGGER.debug("%s", str(theta))
        i = 0
        for file_name in self.files_merged:
            self.files_merged[file_name]['idr'] = lidr
            i += 1
        LOGGER.info("%s", "computing idr done.")

    def write_file(self):
        """
        write output
        """
        for file_name in self.files_merged:
            LOGGER.info("%s", "writing output for " + file_name)
            output_name = PurePath(self.output)\
                .joinpath("idr_" + str(file_name))
            self.files_merged[file_name]\
                .to_csv(output_name,
                        sep='\t',
                        encoding='utf-8',
                        columns=self.column_names + ['idr'],
                        header=False,
                        index=False)
        LOGGER.info("%s", "writing output  done.")


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
        LOGGER.exception("%s", "error: h_function: " + str(err))
        LOGGER.exception("%s", str(cov))
        LOGGER.exception("%s", str(theta))
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
        LOGGER.exception("%s", "error: logLikelihood: " + str(err))
        LOGGER.exception("%s", str(h1_x[i]))
        LOGGER.exception("%s", str(theta))
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
            LOGGER.debug("%s", "warning: EM decreassing logLikelihood rho: " +
                         str(logl_t1 - logl_t0))
            LOGGER.debug("%s", str(theta_t1))
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
    logl_t1 = -np.inf
    u_values = compute_empirical_marginal_cdf(compute_rank(x_score))
    while delta(theta_t0, theta_t1, threshold, logl_t1):
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
        plot_log(log, str(log_name) + "_log.pdf")
        plot_classif(x_score,
                     u_values,
                     z_values,
                     lidr,
                     str(log_name) + "_classif.pdf")
        LOGGER.debug("%s", str(theta_t1))
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


class CleanExit():
    """
    Class to wrap code to have cleaner exits
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            return True
        if exc_type is AssertionError:
            return exc_value
        return exc_type is None


def main():
    """
    body of the idr tool
    """
    with CleanExit():
        try:
            setup_logging(OPTIONS)
            NarrowPeaks(file_merge=OPTIONS.merged,
                        file_names=OPTIONS.files,
                        output=OPTIONS.output,
                        score=OPTIONS.score,
                        threshold=OPTIONS.threshold,
                        merge_function=OPTIONS.merge_function)
        except KeyboardInterrupt:
            print("Shutdown requested...exiting")
            sys.exit(0)
        except AssertionError as err:
            print(err)
            sys.exit(0)


THETA_INIT = {'pi': 0.5,
              'mu': 0.0,
              'sigma': 1.0,
              'rho': 0.9}
LOGGER = logging.getLogger(path.splitext(path.basename(sys.argv[0]))[0])
OPTIONS = parse_args()
if __name__ == "__main__":
    main()
