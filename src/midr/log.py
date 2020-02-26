#!/usr/bin/python3
"""Compute the Irreproducible Discovery Rate (IDR) from NarrowPeaks files

This section of the code provide facilities to handle logs in the mIDR project
"""
import logging
import sys
from os import path

import numpy as np

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
    root = logging.getLogger(__name__)
    root.setLevel(logging.INFO)
    file_handler = logging.FileHandler(options.output + "/log.txt")
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)
    handler_list = [file_handler]
    if options.verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        root.addHandler(console)
        handler_list.append(console)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(message)s", datefmt='%H:%M:%S',
        handlers=handler_list
    )

