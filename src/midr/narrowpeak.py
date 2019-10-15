#!/usr/bin/python3

"""Compute the Irreproducible Discovery Rate (IDR) from NarrowPeaks files

This section of the project provides facilitites to handle NarrowPeaks files
and compute IDR on the choosen value in the NarrowPeaks columns
"""

import sys
from os import path, makedirs, access, R_OK, W_OK
from pathlib import PurePath
from typing import List, Any, Union

import numpy as np
import pandas as pd

import logging

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

import log


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
                "error: file names must be unique (option --file or -f): {}" \
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
            file_path = PurePath(self.file_names[file_name]) \
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
            self.files[file_name] = self.files[file_name] \
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
            chr_a_index = self.files_merged[file_name]['chr'] \
                .unique().tolist().index(peak_a['chr'])
            chr_b_index = self.files[file_name]['chr'] \
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
            chr_b_index = self.files_merged[file_name]['chr'] \
                .unique().tolist().index(peak_b['chr'])
            chr_a_index = self.files[file_name]['chr'] \
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
            output_name = PurePath(self.output) \
                .joinpath("idr_" + str(file_name))
            self.files_merged[file_name] \
                .to_csv(output_name,
                        sep='\t',
                        encoding='utf-8',
                        columns=self.column_names + ['idr'],
                        header=False,
                        index=False)
        LOGGER.info("%s", "writing output  done.")


def narrowpeak_cols() -> list:
    """
    Return list of narrowpeak column names
    :return: a list of string
    """
    return ['chr', 'start', 'stop', 'name', 'score', 'strand',
            'signalValue', 'pValue', 'qValue', 'peak']


def narrowpeaks_score() -> str:
    """
    Return the score column of narrowpeak files
    :return:
    """
    return 'signalValue'


def narrowpeaks_sort_cols() -> list:
    """
    Return a list of column to sort and merge peaks on
    :return: a list of string
    """
    return ['chr', 'start', 'stop', 'strand', 'peak']


def readbed(bed_path: PurePath,
            bed_cols: list = narrowpeak_cols()) -> pd.DataFrame:
    """
    Read a bed file from a PurePath object
    :type bed_cols: list of str
    :param bed_path: PurePath of the bedfile
    :param bed_cols: list of columns names
    :return: a pd.DataFrame corresponding to the bed file
    """
    assert path.isfile(str(bed_path)), f"File {str(bed_path)} doesn't exist"
    assert access(str(bed_path), R_OK), f"File {str(bed_path)} isn't readable"
    return pd.read_csv(
        bed_path,
        sep='\t',
        header=None,
        names=bed_cols
    )


def sort_bed(bed_file: pd.DataFrame,
             sort_cols: list = narrowpeaks_sort_cols()) -> pd.DataFrame:
    """
    Sort bed files according to sort_cols columns
    :param bed_file: bed file loaded as a pd.DataFrame
    :param sort_cols: list of columns to sort the pd.DataFrame on
    :return: None the array is sorted as is
    >>> sort_bed(bed_file=pd.DataFrame({
    ... 'chr': ['b','b', 'a', 'a', 'a', 'a'],
    ... 'start': [100000, 20, 10, 100, 1000, 5],
    ... 'stop': [100100, 40, 15, 150, 2000, 8],
    ... 'strand': ['.', '.', '.', '.', '.', '.'],
    ... 'peak': [100050, 30, 12, 125, 1500, 6]
    ... }))
      chr   start    stop strand    peak
    5   a       5       8      .       6
    2   a      10      15      .      12
    3   a     100     150      .     125
    4   a    1000    2000      .    1500
    1   b      20      40      .      30
    0   b  100000  100100      .  100050
    """
    return bed_file.sort_values(by=sort_cols)


def readbeds(bed_paths: list,
             bed_cols: list = narrowpeak_cols(),
             sort_cols: list = narrowpeaks_sort_cols()) -> list:
    """
    Read a list of bed files from a PurePath list
    :type bed_paths: list of PurePath objects
    :param bed_paths: list of PurePath
    :param bed_cols: list of bedfiles columns
    :param sort_cols: list of columns to sort the pd.DataFrame on
    :return: list of pd.DataFrame
    """
    bed_files: List[PurePath] = list()
    for bed_path in bed_paths:
        bed_files.append(
            sort_bed(
                bed_file=readbed(bed_path, bed_cols),
                sort_cols=sort_cols
            )
        )
    return bed_files


def readfiles(file_names: list,
              file_cols: list = narrowpeak_cols()) -> list:
    """
    Reads a list of bed filenames and return a list of pd.DataFrame
    :param file_names: list of bed files to read
    :param file_cols: list of bed file columns
    :return: list[pd.DataFrame] containing the file csv columns
    """
    bed_paths = list()
    for file_name in file_names:
        bed_paths.append(PurePath(file_name))
    return readbeds(bed_paths=bed_paths, bed_cols=file_cols)


def pos_overlap(pos_ref: pd.Series, pos: pd.Series) -> bool:
    """
    Return True if two bed position overlap with each other
    :param pos_ref bed line in the reference bed file,
    :param pos bed line in the considered bed file
    :return: bool, True if pos overlap and false otherwise

    >>> pos_overlap(pos_ref = pd.Series({'chr': 'a', 'start': 100, 'stop': 120,
    ... 'strand': "."}),
    ... pos = pd.Series({'chr': 'a', 'start': 100, 'stop': 120, 'strand': "."}))
    True
    >>> pos_overlap(pos_ref = pd.Series({'chr': 'a', 'start': 100, 'stop':
    ... 120, 'strand': "."}),
    ... pos = pd.Series({'chr': 'a', 'start': 110, 'stop': 130, 'strand': "."}))
    True
    >>> pos_overlap(pos_ref = pd.Series({'chr': 'a', 'start': 100, 'stop':
    ... 120, 'strand': "."}),
    ... pos = pd.Series({'chr': 'b', 'start': 100, 'stop': 120, 'strand': "."}))
    False
    >>> pos_overlap(pos_ref = pd.Series({'chr': 'a', 'start': 100, 'stop':
    ... 120, 'strand': "."}),
    ... pos = pd.Series({'chr': 'b', 'start': 130, 'stop': 150, 'strand': "."}))
    False
    >>> pos_overlap(pos_ref = pd.Series({'chr': 'a', 'start': 130, 'stop':
    ... 150, 'strand': "."}),
    ... pos = pd.Series({'chr': 'b', 'start': 100, 'stop': 120, 'strand': "."}))
    False
    """
    for pos_col in ['chr', 'strand']:
        assert isinstance(pos_ref[pos_col], str), \
            f'pos_overlapp: {pos_col} = {pos_ref[pos_col]} isn\'t a str'
        assert isinstance(pos[pos_col], str), \
            f'pos_overlapp: {pos_col} = {pos[pos_col]} isn\'t a str'
        if pos_ref[pos_col] != pos[pos_col]:
            return False
    # pos before pos_ref
    if pos_ref['start'] > pos['stop']:
        return False
    # pos after pos_ref
    if pos_ref['stop'] < pos['start']:
        return False
    return True


def best_peak(ref_peak: pd.Series, peaks: pd.DataFrame,
              start_pos: int = 0) -> int:
    """
    Return the index of the closest peak to peak_ref in peaks in case of
    equality return the one with the highest score
    :param ref_peak: the reference peak (line of a narrowpeak file)
    :param peaks: a list of peaks (lines of a narrowpeak file)
    :param: start_pos: int starting position of peaks
    :return: int index of the closest peak in peaks

    >>> best_peak(ref_peak=pd.Series({'peak': 100, 'score': 20}),
    ... peaks=pd.DataFrame({'peak': [90, 110, 105], 'score': [5, 10, 20]}))
    2
    >>> best_peak(ref_peak=pd.Series({'peak': 100, 'score': 20}),
    ... peaks=pd.DataFrame({'peak': [90, 105, 105], 'score': [5, 20, 10]}))
    1
    """
    pos = abs(peaks.peak - ref_peak.peak).idxmin()
    if peaks.peak.where(peaks.iloc[pos].peak == peaks.peak).size == 1:
        return pos + start_pos
    else:
        return peaks.score.where(peaks.iloc[pos].peak == peaks.peak).idxmax() \
               + start_pos


def merge_peak(ref_peak: pd.Series, peak: pd.Series,
               pos_cols: list = narrowpeaks_sort_cols()) -> pd.Series:
    """
    Return merged peaks between position of ref_peak and everythings else
    from peak
    :param ref_peak: line of ref_peaks narrowpeak
    :param peak: line of peak narrowpeak
    :param pos_cols: list list of columns name for position information
    :return: line of narrowpeak

    >>> merge_peak(
    ... ref_peak=pd.Series({'chr': 'a', 'start': 100, 'stop': 120,
    ... 'strand': ".", 'peak': 100, 'score': 20}),
    ... peak=pd.Series({'chr': 'a', 'start': 200, 'stop': 220,
    ... 'strand': ".", 'peak': 140, 'score': 45})
    ... )
    chr         a
    start     100
    stop      120
    strand      .
    peak      100
    score      45
    dtype: object
    """
    merged_peak = peak.copy()
    for pos_col in pos_cols:
        merged_peak[pos_col] = ref_peak[pos_col]
    return merged_peak


def iter_peaks(merged_peaks: pd.DataFrame, peaks: pd.DataFrame, merged_peaks_it,
               peaks_it) -> (int, int):
    """
    Move iterator over ref_peaks and peaks for the merge_peaks() function
    :param merged_peaks: pd.DataFrame of the reference peaks
    :param peaks: pd.DataFrame of the peaks we want to merge
    :param merged_peaks_it: iterator over row index in the ref_peaks
    pd.DataFrame
    :param peaks_it: iterator over row index in the peaks pd.DataFrame
    :yield: (merged_peak, peak - 1, prev_peak) triplet of positions

    >>> test_iter = iter_peaks(
    ... merged_peaks=pd.DataFrame({
    ... 'chr': ['a', 'a', 'a', 'a', 'a'],
    ... 'start': [100, 1000, 4000, 100000, 200000],
    ... 'stop': [500, 3000, 10000, 110000, 230000],
    ... 'strand': [".", ".", ".", ".", "."],
    ... 'peak': [250, 2000, 7000, 100000, 215000],
    ... 'score': [20, 100, 15, 30, 200]}),
    ... peaks=pd.DataFrame({
    ... 'chr': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
    ... 'start': [100, 100, 1000, 4000, 4000, 4000, 100000, 200000, 200000,
    ... 200000],
    ... 'stop': [500, 500, 3000, 10000, 10000, 10000, 110000, 230000, 230000,
    ... 230000],
    ... 'strand': [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ... 'peak': [250, 200, 2000, 5000, 6000, 7000, 100000, 205000, 215000,
    ... 220000],
    ... 'score': [20, 15, 100, 15, 30, 14, 30, 200, 300, 400]}),
    ... merged_peaks_it=iter(range(5)),
    ... peaks_it=iter(range(10))
    ... )
    >>> next(test_iter)
    (0, 1, 0)
    >>> next(test_iter)
    (1, 2, 2)
    >>> next(test_iter)
    (2, 5, 3)
    >>> next(test_iter)
    (3, 6, 6)
    >>> next(test_iter)
    (4, 9 ,7)
    >>> try:
    ...     next(test_iter)
    ... except StopIteration:
    ...     print("end")
    end
    """
    merged_peak = next(merged_peaks_it)
    peak = next(peaks_it)
    prev_peak = None
    while True:
        try:
            # if merged_peak before peak
            if peaks.iloc[peak]['stop'] < merged_peaks.iloc[
                merged_peak]['start'
            ]:
                peak = next(peaks_it)
                prev_peak = None
            # if merged_peak after peak
            if merged_peaks.iloc[merged_peak]['stop'] < peaks.iloc[
                peak]['start'
            ]:
                merged_peak = next(merged_peaks_it)
                prev_peak = None
            if pos_overlap(
                    pos_ref=merged_peaks.iloc[merged_peak],
                    pos=peaks.iloc[peak]
            ):
                if prev_peak is None:
                    prev_peak = peak
                peak = next(peaks_it)
        except StopIteration:
            break  # Iterator exhausted: stop the loop
        else:
            if not pos_overlap(
                    pos_ref=merged_peaks.iloc[merged_peak],
                    pos=peaks.iloc[peak]
            ) and prev_peak is not None:
                yield ((merged_peak, peak - 1, prev_peak))
                prev_peak = None



def merge_peaks(ref_peaks: pd.DataFrame,
                peaks: pd.DataFrame,
                score: str = narrowpeaks_score(),
                pos_cols: list = narrowpeaks_sort_cols()) -> pd.DataFrame:
    """
    Copy peaks values from peaks into the corresponding position in merged_peaks
    Peaks not found in peaks have a score of nan
    :param ref_peaks: pd.DataFrame which is a copy of ref_peaks
    :param peaks: pd.DataFrame of the peaks we want to merge
    :param score: str with the name of the score column
    :param pos_cols: list list of columns name for position information
    :return: pd.DataFrame of the merged peaks
    """
    merged_peaks = ref_peaks.copy()
    merged_peaks_it = iter(range(len(merged_peaks)))
    peaks_it = iter(range(len(peaks)))
    for merged_peak, peak, prev_peak in iter_peaks(
            merged_peaks=merged_peaks,
            peaks=peaks,
            merged_peaks_it=merged_peaks_it,
            peaks_it=peaks_it):
        merged_peaks.iloc[merged_peak, :] = merged_peak(
            ref_peaks=merged_peaks.iloc[merged_peak, :],
            peak=best_peak(
                peak_ref=merged_peaks.iloc[merged_peak, :],
                peaks=peaks.iloc[prev_peak:peak, :],
                start_pos=prev_peak
            ),
            pos_cols=pos_cols
        )
    return merged_peaks


def narrowpeaks2array(np_list: list,
                      score_col: str = narrowpeaks_score()) -> np.array:
    """
    convert a list of pd.DataFrame representing bed files to an np.array
    of their score column
    :type np_list: list[pd.DataFrame]
    :type score_col: str colname of the score column
    :param np_list: list of pd.DataFrame representing bed files
    :param score_col: score column to use to compute IDR
    :return np.array whose columns are the score columns of the bed files

    >>> narrowpeaks2array(np_list=[
    ... pd.DataFrame({'peak': [90, 105, 105], 'signalValue': [5, 20, 10]}),
    ... pd.DataFrame({'peak': [90, 105, 105], 'signalValue': [5, 21, 11]}),
    ... pd.DataFrame({'peak': [90, 105, 105], 'signalValue': [5, 22, 12]})]
    ... )
    array([[ 5,  5,  5],
           [20, 21, 22],
           [10, 11, 12]])
    """
    scores = list()
    np_file: pd.DataFrame
    for np_file in np_list:
        scores.append(np.array(np_file[score_col].to_numpy()))
    return np.stack(scores, axis=-1)


LOGGER = logging.getLogger(path.splitext(path.basename(sys.argv[0]))[0])

if __name__ == "__main__":
    import doctest

    doctest.testmod()
