#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. module:: __main__
    :platform: Unix
    :synopsis: the top-level module of ImLine that contains the entry point and handles built-in commands.

.. moduleauthor:: Cem Baybars GÜÇLÜ <cem.baybars@gmail.com>
"""

import argparse
import os
import sys

from imline.logging import LogManager

import imline.__init__
from imline import dot_imline_dir

logger = None


def start(args):
    """Function that starts the ImLine with the correct mode according to command-line arguments.

    Args:
        args:       Command-line arguments.
    """

    try:
        pass

    except KeyboardInterrupt:
        logger.debug("Keyboard Interruption")


def start_sub(args):
    """Function that starts the ImLine with the sub jobs according to command-line arguments.

    Args:
        args:       Command-line arguments.
    """

    if args["sub_jobs"] == "multiply-dataset":
        from imline.multiplication import multiply_dataset

        multiply_dataset(args["dataset"], args["method"], args["output"])

    elif args["sub_jobs"] == "mark-object":
        from imline.object_detection import mark_object

        mark_object(args["dataset"], args["output"])

    elif args["sub_jobs"] == "mark-key-points":
        from imline.mapping import Mapper
        imline.mapper = Mapper(args)

        imline.mapper.start_by(raw_dataset=args["raw_dataset"])

    elif args["sub_jobs"] == "create-dataset":
        from imline.downloading import ImageDownloader

        image_downloader = ImageDownloader()

        image_downloader.download(args["query"], args["format"], args["limit"], args["print_urls"], args["size"], args["aspect_ratio"], args["output"])

    elif args["sub_jobs"] == "create-maps":

        from imline.mapping import Mapper
        imline.mapper = Mapper(args)

        if args["key_point_marker"] == "stasm":
            imline.mapper.start_by_stasm(dataset=args["dataset"])


def prepare(args):
    """The function that prepares the working environment for storing data during running.

    Args:
        args:       Command-line arguments.
    """
    from imline.presentation import startup_banner
    startup_banner()

    if not os.path.exists(dot_imline_dir):
        os.mkdir(dot_imline_dir)

    imline.log_manager = LogManager(args)

    global logger
    logger = imline.log_manager.get_logger(__name__, "DEBUG")
    logger.info("Logger integration successful.")

    if args["sub_jobs"]:
        start_sub(args)
        sys.exit(1)


def initiate():
    """The top-level method to serve as the entry point of ImLine.

    This method is the entry point defined in `setup.py` for the `imline` executable that placed a directory in `$PATH`.

    This method parses the command-line arguments and handles the top-level initiations accordingly.
    """

    ap = argparse.ArgumentParser()

    other_gr = ap.add_argument_group('Others')
    other_gr.add_argument("--ripe-dataset", help="Images folder that keeps already marked images and their key-points\"s JSON data.", type=str)
    other_gr.add_argument("--environment", help="The running environment. It specify the configuration files and logs. To use: either `production`, `development` or `testing`. Default is production", action="store", type=str, choices=["production", "development", "testing"], default="production")
    other_gr.add_argument("-v", "--verbose", help="Print various debugging logs to console for debug problems", action="store_true")
    other_gr.add_argument("--version", help="Display the version number of ImLine.", action="store_true")

    sub_p = ap.add_subparsers(dest="sub_jobs", help="officiate the sub-jobs")  # if sub-commands not used their arguments create raise.

    ap_object = sub_p.add_parser("multiply-dataset", help="Mark found objects of the given images to the ImLine.")
    ap_object.add_argument("-d", "--dataset", help="Images folder that will be marked for detecting object.", type=str)
    ap_object.add_argument("--method", help="Multiplying method for the images. Either `rotate`, `resize` or `shift`", type=str, default="shift", choices=["rotate", "resize", "shift"])
    ap_object.add_argument("--output", help="Data folder path that will be keep key-point coordinates and mapped image files.", type=str)

    ap_object = sub_p.add_parser("mark-object", help="Mark found objects of the given images to the ImLine.")
    ap_object.add_argument("-d", "--dataset", help="Images folder that will be marked for detecting object.", type=str)
    ap_object.add_argument("--output", help="Data folder path that will be keep key-point coordinates and mapped image files.", type=str)

    ap_key_po = sub_p.add_parser("mark-key-points", help="Mark key-points of the given images to the ImLine.")
    ap_key_po.add_argument("--raw-dataset", help="Images folder that will be marked for specifying key-points.", type=str)
    ap_key_po.add_argument("--output", help="Data folder path that will be keep key-point coordinates and mapped image files.", type=str)

    ap_cre_dataset = sub_p.add_parser("create-dataset", help="Create a dataset that become images that fetched from Google with given `query` parameter")
    ap_cre_dataset.add_argument("--query", help="Context or description of the images that downloaded in dataset.", type=str, required=True)
    ap_cre_dataset.add_argument("--format", help="Format of the images that will downloaded.", type=str, default="jpg")
    ap_cre_dataset.add_argument("--limit", help="The number of images to be downloaded.", type=int, default=5)
    ap_cre_dataset.add_argument("--print-urls", help="Flag to print the image file url.", type=bool, default=True)
    ap_cre_dataset.add_argument("--size", help="the image size which can be specified manually either `large`, `medium`, `icon`", type=str, default="medium", choices=["large", "medium", "icon"])
    ap_cre_dataset.add_argument("--aspect-ratio", help="Denotes the height width ratio of images to download. `tall`, `square`, `wide`, `panoramic`", type=str, default="square", choices=["tall", "square", "wide", "panoramic"])
    ap_cre_dataset.add_argument("--output", help="Data folder path that will be keep downloaded images.", type=str, required=True)

    ap_cre_maps = sub_p.add_parser("create-maps", help="Create maps via key-points of the given images to the ImLine.")
    ap_cre_maps.add_argument("-d", "--dataset", help="Images folder that will be marked for specifying key-points.", type=str, required=True)
    ap_cre_maps.add_argument("--label", help="Label of the given dataset", type=str, default=1)
    ap_cre_maps.add_argument("--draw-maps", help="Flag that allow map drawing on image and save it", action="store_true")
    ap_cre_maps.add_argument("--key-point-marker", help="Marking method that determine key-points on images. to use: either `stasm`", type=str, default="stasm", choices=["stasm"])
    ap_cre_maps.add_argument("--point-link-type", help="Type of combining the points. to use: either `straight`, `delanuay`", type=str, default="point_to_points", choices=["straight", "point_to_points", "delanuay"])
    ap_cre_maps.add_argument("--output", help="Data folder path that will be keep key-point coordinates and mapped image files.", type=str)

    args = vars(ap.parse_args())

    if args["version"]:
        from imline.presentation import versions_banner
        versions_banner()
        sys.exit(1)

    prepare(args)
    start(args)


if __name__ == '__main__':
    initiate()
