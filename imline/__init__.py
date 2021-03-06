#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. module:: __init__
    :platform: Unix
    :synopsis: the top-level module of ImLine that contains the initial module imports and global variables.

.. moduleauthor:: Cem Baybars GÜÇLÜ <cem.baybars@gmail.com>
"""

import os  # Miscellaneous operating system interfaces
import inspect  # Inspect live objects

from os.path import expanduser  # Common pathname manipulations

IMLINE_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

home = expanduser("~")
dot_imline_dir = f'{home}/.imline'

log_manager = None
mapper = None

__author__ = 'Cem Baybars GÜÇLÜ'
__email__ = 'cem.baybars@gmail.com'
__version__ = '0.0.3'
