#!/usr/bin/python3

"""Compute the Irreproducible Discovery Rate (IDR) from NarrowPeaks files

This section of the project provides facilitites to handle NarrowPeaks files
and compute IDR on the choosen value in the NarrowPeaks columns
"""

from os import path, access, R_OK
from pathlib import PurePath
from typing import List
from typing import Callable
import numpy as np
import pandas as pd


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
              file_cols: list = narrowpeak_cols(),
              score_cols: str = narrowpeaks_score(),
              pos_cols: list = narrowpeaks_sort_cols()) -> list:
    """
    Reads a list of bed filenames and return a list of pd.DataFrame
    :rtype: list[pd.DataFrame]
    :param file_names: list of bed files to read
    :param file_cols: list of bed file columns
    :param score_cols: column name of the score to use
    :param pos_cols: list of position column name to sort and merge on
    :return: list[pd.DataFrame] containing the file csv columns
    """
    bed_paths = list()
    for file_name in file_names:
        bed_paths.append(PurePath(file_name))
    return merge_beds(
        bed_files=readbeds(bed_paths=bed_paths, bed_cols=file_cols),
        score_col=score_cols,
        pos_cols=pos_cols
    )


def writefiles(bed_files: list,
               file_names: list,
               idr: np.array,
               outdir: str = "results/"):
    """
    Write output of IDR computation
    :param bed_files: list of bed files (pd.DataFrame)
    :param file_names: list of files names (str)
    :param idr: np.array with local IDR score (columns correspond to bed files)
    :param outdir: output directory
    :return: nothing
    """
    idr_col = 0
    for bed in bed_files:
        output_name = PurePath(outdir).joinpath(
            "idr_" + str(file_names[idr_col])
        )
        bed.assign(idr=idr[idr_col]).to_csv(
            output_name, sep='\t',
            encoding='utf-8',
            header=False,
            index=False
        )


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
              start_pos: int = 0,
              score_col: str = narrowpeaks_score()) -> int:
    """
    Return the index of the closest peak to peak_ref in peaks in case of
    equality return the one with the highest score
    :param ref_peak: the reference peak (line of a narrowpeak file)
    :param peaks: a list of peaks (lines of a narrowpeak file)
    :param start_pos: int starting position of peaks
    :param score_col: str name of the score column
    :return: int index of the closest peak in peaks

    >>> best_peak(ref_peak=pd.Series({'peak': 100, 'signalValue': 20}),
    ... peaks=pd.DataFrame({'peak': [90, 110, 105],
    ... 'signalValue': [5, 10, 20]}))
    2
    >>> best_peak(ref_peak=pd.Series({'peak': 100, 'signalValue': 20}),
    ... peaks=pd.DataFrame({'peak': [90, 105, 105],
    ... 'signalValue': [5, 20, 10]}))
    1
    >>> test_peak = pd.DataFrame({'peak': [90, 105, 105, 90, 105, 105],
    ... 'signalValue': [5, 20, 10, 5, 20, 10]})
    >>> best_peak(ref_peak=pd.Series({'peak': 100, 'signalValue': 20}),
    ... peaks=test_peak.iloc[3:6, :])
    1
    """
    if peaks.shape[0] == 1:
        return start_pos
    peaks = peaks.reset_index()
    pos = abs(peaks.peak - ref_peak.peak).idxmin()
    if peaks.peak.where(peaks.iloc[pos].peak == peaks.peak).size == 1:
        return pos + start_pos
    else:
        return peaks[score_col] \
                   .where(
            peaks.iloc[pos].peak == peaks.peak).idxmax() + start_pos


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
    (0, 2, 0)
    >>> next(test_iter)
    (1, 3, 2)
    >>> next(test_iter)
    (2, 6, 3)
    >>> next(test_iter)
    (3, 7, 6)
    >>> next(test_iter)
    (4, 10, 7)
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
                    merged_peak]['start']:
                peak = next(peaks_it)
                prev_peak = None
            # if merged_peak after peak
            if merged_peaks.iloc[merged_peak]['stop'] < peaks.iloc[
                    peak]['start']:
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
            if prev_peak is None:
                prev_peak = peak
            yield ((merged_peak, peak + 1, prev_peak))
            break  # Iterator exhausted: stop the loop
        else:
            if not pos_overlap(
                    pos_ref=merged_peaks.iloc[merged_peak],
                    pos=peaks.iloc[peak]
            ) and prev_peak is not None:
                yield ((merged_peak, peak, prev_peak))
                prev_peak = None


def merge_peaks(ref_peaks: pd.DataFrame,
                peaks: pd.DataFrame,
                score_col: str = narrowpeaks_score(),
                pos_cols: list = narrowpeaks_sort_cols()) -> pd.DataFrame:
    """
    Copy peaks values from peaks into the corresponding position in merged_peaks
    Peaks not found in peaks have a score of nan
    :param ref_peaks: pd.DataFrame which is a copy of ref_peaks
    :param peaks: pd.DataFrame of the peaks we want to merge
    :param score_col: str with the name of the score column
    :param pos_cols: list list of columns name for position information
    :return: pd.DataFrame of the merged peaks

    >>> merge_peaks(
    ... ref_peaks=pd.DataFrame({
    ... 'chr': ['a', 'a', 'a', 'a', 'a', 'a'],
    ... 'start': [50, 100, 1000, 4000, 100000, 200000],
    ... 'stop': [60, 500, 3000, 10000, 110000, 230000],
    ... 'strand': [".", ".", ".", ".", ".", "."],
    ... 'peak': [55, 250, 2000, 7000, 100000, 215000],
    ... 'signalValue': [10, 20, 100, 15, 30, 200]}),
    ... peaks=pd.DataFrame({
    ... 'chr': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
    ... 'start': [100, 100, 1000, 4000, 4000, 4000, 100000, 200000, 200000,
    ... 200000],
    ... 'stop': [500, 500, 3000, 10000, 10000, 10000, 110000, 230000, 230000,
    ... 230000],
    ... 'strand': [".", ".", ".", ".", ".", ".", ".", ".", ".", "."],
    ... 'peak': [250, 200, 2000, 5000, 6000, 7000, 100000, 205000, 215000,
    ... 220000],
    ... 'signalValue': [20, 15, 100, 15, 30, 14, 30, 200, 300, 400]})
    ... )
      chr   start    stop strand    peak  signalValue
    0   a      50      60      .      55          NaN
    1   a     100     500      .     250         20.0
    2   a    1000    3000      .    2000        100.0
    3   a    4000   10000      .    7000         14.0
    4   a  100000  110000      .  100000         30.0
    5   a  200000  230000      .  215000        300.0
    """
    merged_peaks = ref_peaks.copy()
    merged_peaks[merged_peaks.columns.difference(pos_cols)] = np.NaN
    merged_peaks_it = iter(range(len(merged_peaks)))
    peaks_it = iter(range(len(peaks)))
    for merged_peak, peak, prev_peak in iter_peaks(
            merged_peaks=merged_peaks,
            peaks=peaks,
            merged_peaks_it=merged_peaks_it,
            peaks_it=peaks_it):
        if prev_peak != peak:
            peak = best_peak(
                ref_peak=merged_peaks.iloc[merged_peak],
                peaks=peaks.iloc[prev_peak:peak],
                start_pos=prev_peak,
                score_col=score_col
            )
        merged_peaks.iloc[merged_peak, :] = merge_peak(
            ref_peak=merged_peaks.iloc[merged_peak],
            peak=peaks.iloc[peak],
            pos_cols=pos_cols
        )
    return merged_peaks


def merge_beds(bed_files: list, ref_pos=0,
               score_col: str = narrowpeaks_score(),
               pos_cols: list = narrowpeaks_sort_cols()) -> list:
    """
    Merge a list of bed according to position in a reference in the list
    :param bed_files: list of pd.DataFrame representing bed files
    :param ref_pos: position of the reference bed in the bed_files list
    :param score_col: str with the name of the score column
    :param pos_cols: list list of columns name for position information
    :return: a list of bed files (pd.DataFrame)
    """
    merged_files = list()
    nan_pos = list()
    assert ref_pos < len(bed_files), \
        "error: ref_pos must be in the list of bed"
    for bed in bed_files:
        if bed is not bed_files[ref_pos]:
            merged_files.append(
                merge_peaks(
                    ref_peaks=bed_files[ref_pos],
                    peaks=bed,
                    score_col=score_col,
                    pos_cols=pos_cols
                )
            )
            nan_pos = nan_pos + list(
                merged_files[-1].index[
                    merged_files[-1][score_col].apply(np.isnan)
                ].to_numpy()
            )
    nan_pos = set(nan_pos)
    for merged in range(len(merged_files)):
        merged_files[merged] = merged_files[merged].drop(nan_pos)
    return merged_files


def narrowpeaks2array(np_list: list,
                      score_col: str = narrowpeaks_score()) -> np.array:
    """
    convert a list of pd.DataFrame representing bed files to an np.array of
    their score column
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
    scores = np.stack(scores, axis=-1)
    return scores


def process_bed(file_names: list,
                outdir: str,
                idr_func: Callable[[np.array], np.array],
                file_cols: list = narrowpeak_cols(),
                score_cols: str = narrowpeaks_score(),
                pos_cols: list = narrowpeaks_sort_cols()):
    """
    Process a list of bed files names with the first names the merged bed files
    :param file_names: list of files path
    :param outdir: output directory
    :param idr_func: idr function to apply
    :param file_cols: list of bed file columns
    :param score_cols: column name of the score to use
    :param pos_cols: list of position column name to sort and merge on
    :return: nothing
    """
    bed_files = readfiles(
        file_names=file_names,
        file_cols=file_cols,
        score_cols=score_cols,
        pos_cols=pos_cols,
    )
    theta, local_idr = idr_func(
       narrowpeaks2array(
           np_list=bed_files,
           score_col=score_cols
       )
    )
    print(theta)
    writefiles(
        bed_files=bed_files,
        file_names=file_names,
        idr=local_idr,
        outdir=outdir
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
