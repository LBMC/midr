#!/usr/bin/python3
"""Compute the Irreproducible Discovery Rate (IDR) from NarrowPeaks files

This section of the code provide facilities to handle logs in the mIDR project
"""
import logging
import sys
from os import path

import numpy as np
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


LOGGER = logging.getLogger(path.splitext(path.basename(sys.argv[0]))[0])
