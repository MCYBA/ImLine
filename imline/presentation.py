#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. module:: presentation
    :platform: Unix
    :synopsis: the top-level submodule of ImLine that contains the classes related to ImLine's presenting itself to user ability.

.. moduleauthor:: Cem Baybars GÜÇLÜ <cem.baybars@gmail.com>
"""

import shutil  # High-level file operations
import pkg_resources

from subprocess import call  # Subprocess managements


def startup_banner():
    """The top-level method to create startup banner of Imline itself.
    """

    (columns, lines) = shutil.get_terminal_size()

    call(f'figlet -f smslant \'ImLine\' | boxes -d scroll -a hcvc -p h8 | /usr/games/lolcat -a -d 1', shell=True)  # covers 65 columns.

    call(f'echo {int(columns * 0.85) * "_"} | /usr/games/lolcat', shell=True)
    print("\n")


def versions_banner():
    """The top-level method to draw banner for showing versions of Imline.
    """

    import imline.__init__
    from imline.logging import LogManager

    imline.log_manager = LogManager(args={"verbose": False, "environment": None})

    imline_version = pkg_resources.get_distribution("imline").version

    versions = f'imline: {imline_version}'
    call(f'figlet -f term \'{versions}\' | boxes -d spring -a hcvc -p h8 | /usr/games/lolcat -a -d 1', shell=True)
