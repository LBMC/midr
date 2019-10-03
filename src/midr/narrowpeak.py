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

import sys
from os import path, makedirs, access, R_OK, W_OK
from pathlib import PurePath
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def parse_args(args):
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

    def __init__(self, params, idr_func):
        """
        Create narrowpeak DataFrame
        """
        self.files = dict()
        self.files_merged = dict()
        if params.merge_function == 'mean':
            self.merge_function = np.mean
        elif params.merge_function == 'median':
            self.merge_function = np.median
        elif params.merge_function == 'min':
            self.merge_function = min
        else:
            self.merge_function = max
        if params.score in self.column_names[6:9]:
            self.score = params.score
        else:
            LOGGER.exception("%s", "error: " + str(params.score) +
                             " must be a NarrowPeak score column " +
                             str(self.score_columns))
            quit(-1)
        file_path = PurePath(params.merged)
        self.file_merge = file_path.name
        self.file_merge_path = file_path.parent
        self.file_names = dict()
        for full_path in params.files:
            file_path = PurePath(full_path)
            assert file_path.name not in self.file_names, \
                "error: file names must be unique (option --file or -f): {}"\
                .format(file_path.name)
            self.file_names[file_path.name] = file_path.parent
        self.output = PurePath(params.output)
        self.threshold = params.threshold
        self.idr_func = idr_func
        self.run_analysis()

    def run_analysis(self):
        """
        run midr analysis
        """
        self.read_peaks()
        self.sort_peaks()
        self.collapse_peaks()
        self.merge_peaks()
        assert access(PurePath(self.output).parent, W_OK), \
            "Folder {} isn't writable".format(self.output)
        if not path.isdir(self.output):
            makedirs(self.output)
        self.idr()
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

    def idr(self):
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
            pval = np.array(self.files_merged[file_name]['qValue'])
            data[:, i] = score.astype(float) + pval.astype(float)
            i += 1
        theta, lidr = self.idr_func(x_score=data,
                                    threshold=self.threshold,
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





def narrowpeaks2array(np_list: pd.array, score_col: str) -> np.array:
    """
    convert a list of pd.array representing narrowpeak files to an np.array
    of their score column
    """
    scores = None
    for np_file in np_list:
        if scores is None:
            scores = np.array(np_file.loc[score_col])
        else:
            np.append(scores, np_file.loc[score_col], axis=1)
    return scores


LOGGER = logging.getLogger(path.splitext(path.basename(sys.argv[0]))[0])
OPTIONS = parse_args(args=sys.argv[1:])
