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
import argparse
from os import path, access, W_OK, makedirs
from pathlib import PurePath

import midr.narrowpeak as narrowpeak
import midr.raw_matrix as raw_matrix
import midr.log as log
import midr.idr as idr
import midr.samic as samic


class CustomFormatter(argparse.RawDescriptionHelpFormatter,
                      argparse.ArgumentDefaultsHelpFormatter):
    """
    helper class to make ArgumentParser
    """


def parse_args(args):
    """Parse arguments."""
    parser = argparse.ArgumentParser(
        description=sys.modules[__name__].__doc__,
        formatter_class=CustomFormatter
    )

    arg = parser.add_argument_group("IDR settings")
    arg.add_argument("--merged", "-m", metavar="FILE",
                     dest='merged',
                     required=False,
                     default=argparse.SUPPRESS,
                     type=str,
                     help="file of the merged NarrowPeaks")
    arg.add_argument("--files", "-f", metavar="FILES",
                     dest='files',
                     required=False,
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
                     help="Threshold value for the precision of the estimators")
    arg.add_argument("--merge_function", "-mf", metavar="MERGE_FUNCTION",
                     dest='merge_function',
                     required=False,
                     default='max',
                     type=str,
                     help="function to determine the score to keep for \
                     overlapping peak within a replica ('sum', 'max', "
                          "'mean', \
                    'median', 'min')")
    arg.add_argument("--size", "-ws", metavar="SIZE_MERGE",
                     dest='size_merge',
                     required=False,
                     default=100,
                     type=int,
                     help="distance (bp) to add before and after each peak "
                          "before \
                     merging finding match between --merged file and --files \
                     files")
    arg.add_argument("--nodrop", "-nd", action="store_false",
                     dest="drop_unmatched",
                     required=False,
                     default=True,
                     help="don't drop peak unmatched in any bed. The score of \
                     the absent peak is set to 0.0")
    arg.add_argument("--method", "-mt", metavar="METHOD",
                     dest='method',
                     required=False,
                     default="archimedean",
                     type=str,
                     help="copula model to use('archimedean' or 'gaussian'")
    arg.add_argument("--cpu", "-cpu", metavar="CPU",
                     dest='cpu',
                     required=False,
                     default=1,
                     type=int,
                     help="number of thread to use for merging the beds files")
    arg.add_argument("--debug", "-d", action="store_true",
                     default=False,
                     help="enable debugging")
    arg.add_argument("--verbose", "-v", action="store_true",
                     default=False,
                     help="log to console")
    arg.add_argument("--matrix", metavar="FILE",
                     dest='matrix',
                     required=False,
                     default=argparse.SUPPRESS,
                     type=str,
                     help="matrix file of the peaks score in raw (tsv "
                          "format), replace the --merge and --files options "
                          "if used")
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


def main(options=parse_args(args=sys.argv[1:])):
    """
    body of the idr tool
    """
    with CleanExit():
        try:
            assert access(PurePath(options.output).parent, W_OK), \
                "Folder {} isn't writable".format(options.output)
            if not path.isdir(options.output):
                makedirs(options.output)
            assert ("merged" in vars(options).keys() and "files" in
                    vars(options).keys()
                    ) or "matrix" in vars(options).keys(), \
                "must either provide a --merged file and a list of --files, " \
                "or a --matrix file"
            log.setup_logging(options)
            model = samic.samic
            if options.method == 'gaussian':
                model = idr.pseudo_likelihood
            if "merged" in vars(options).keys() and \
                 "files" in vars(options).keys():
                narrowpeak.process_bed(
                    file_names=[options.merged] + options.files,
                    outdir=options.output,
                    idr_func=model,
                    size=options.size_merge,
                    merge_function=options.merge_function,
                    score_cols=options.score,
                    threshold=options.threshold,
                    file_cols=narrowpeak.narrowpeaks_cols(),
                    pos_cols=narrowpeak.narrowpeaks_sort_cols(),
                    drop_unmatched=options.drop_unmatched,
                    thread_num=options.cpu
                )
            elif "matrix" in vars(options).keys():
                raw_matrix.process_matrix(
                    file_name=options.matrix,
                    outdir=options.output,
                    idr_func=model
                )
        except KeyboardInterrupt:
            print("Shutdown requested...exiting")
            sys.exit(0)
        except AssertionError as err:
            print(err)
            sys.exit(0)


if __name__ == "__main__":
    main(options=parse_args(args=sys.argv[1:]))
