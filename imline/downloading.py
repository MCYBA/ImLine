#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. module:: downloading
    :platform: Unix
    :synopsis: the top-level submodule of ImLine that contains the methods and classes related to ImLine's ability that is collecting images form google index by keywords.

.. moduleauthor:: Cem Baybars GÜÇLÜ <cem.baybars@gmail.com>
"""
from google_images_download import google_images_download

from imline import dot_imline_dir
from imline import log_manager

logger = log_manager.get_logger(__name__, "DEBUG")


class ImageDownloader:
    """Class to define an image downloader that is for creating dataset via downloading Google searched images.

    This class provides necessary initiations and functions named :func:`imline.downloading.ImageDownloader.start`
    for creating a Record object and start recording by this object.
    """

    def __init__(self, ):
        """Initialization method of :class:`t_system.downloading.ImageDownloader` class.
        """

        self.response = google_images_download.googleimagesdownload()

    def download(self, query, i_format, limit, print_urls, size, aspect_ratio, out_folder):
        """Method to start mapping of given dataset. If raw dataset given, first start the specifying key-points.

        Args:
            query:                      Context or description of the images that downloaded in dataset.
            i_format:                   Format of the images that will downloaded.
            limit:                      The number of images to be downloaded.
            print_urls:                 Flag to print the image file url..
            size:                       the image size which can be specified manually.
            aspect_ratio:               Denotes the height width ratio of images to download.
            out_folder:                 Data folder path that will be keep downloaded images.
        """
        chrome_driver = "/usr/lib/chromium-browser/chromedriver"

        arguments = {"keywords": query, "format": i_format, "limit": limit, "print_urls": print_urls, "size": size, "aspect_ratio": aspect_ratio, "image_directory": out_folder, "chromedriver": chrome_driver}
        try:
            self.response.download(arguments)
        except FileNotFoundError:
            arguments = {"keywords": query, "format": i_format, "limit": limit, "print_urls": print_urls, "size": size, "image_directory": out_folder, "chromedriver": chrome_driver}
            try:
                self.response.download(arguments)
            except Exception as e:
                logger.error(e)
