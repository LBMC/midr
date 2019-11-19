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
import logging
import sys
import argparse

import midr.narrowpeak as narrowpeak
import midr.log as log
import midr.idr as idr


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
                     default=0.0001,
                     type=float,
                     help="Threshold value for the precision of the estimator")
    arg.add_argument("--merge_function", "-mf", metavar="MERGE_FUNCTION",
                     dest='merge_function',
                     required=False,
                     default='sum',
                     type=str,
                     help="function to determine the score to keep for \
                     overlpping peak ('sum', 'max', 'mean', 'median', 'min')")
    arg.add_argument("--size", "-ws", metavar="SIZE_MERGE",
                     dest='size_merge',
                     required=False,
                     default=100,
                     type=int,
                     help="distante to add before and after each peak before "
                          "merging")
    arg.add_argument("--debug", "-d", action="store_true",
                     default=False,
                     help="enable debugging")
    arg.add_argument("--verbose", "-v", action="store_true",
                     default=False,
                     help="log to console")
    return parser.parse_args(args)


class CleanExit:
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
            log.setup_logging(OPTIONS)
            narrowpeak.process_bed(
                file_names=[OPTIONS.merged] + OPTIONS.files,
                outdir=OPTIONS.output,
                idr_func=idr.pseudo_likelihood,
                size=OPTIONS.size_merge,
                merge_function=OPTIONS.merge_function,
                score_cols=OPTIONS.score,
                threshold=OPTIONS.threshold,
                file_cols=narrowpeak.narrowpeaks_cols(),
                pos_cols=narrowpeak.narrowpeaks_sort_cols()
            )
        except KeyboardInterrupt:
            print("Shutdown requested...exiting")
            sys.exit(0)
        except AssertionError as err:
            print(err)
            sys.exit(0)


OPTIONS = parse_args(args=sys.argv[1:])

if __name__ == "__main__":
    main()
