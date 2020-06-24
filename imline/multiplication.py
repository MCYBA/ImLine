#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. module:: pointing
    :platform: Unix
    :synopsis: the top-level submodule of ImLine that contains the methods and classes related to ImLine's ability that is multiplying the image dataset by transforming its images.

.. moduleauthor:: Cem Baybars GÜÇLÜ <cem.baybars@gmail.com>
"""

import cv2
import os  # Miscellaneous operating system interfaces
import numpy

from imutils import paths, rotate_bound
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from imline import IMLINE_PATH, dot_imline_dir
from imline import log_manager

logger = log_manager.get_logger(__name__, "DEBUG")


def multiply_dataset(dataset, method, output):
    """Method to multiplying given dataset with transforming its images.

    Args:
        dataset:                Image dataset folder that contains only unprocessed images.
        method:                 Multiplying method for the images. Either `rotate`, `resize` or `shift`
        output:                 Image folder that contains processed images.
    """

    image_paths = list(paths.list_images(dataset))

    for (i, image_path) in enumerate(image_paths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))

        if method == "rotate":
            rotate_image(image_path, output)
        elif method == "resize":
            resize_image(image_path, output)
        elif method == "shift":
            shift_image(image_path, output)


def rotate_image(image_path, output):
    """Method to rotating image that in given image_path.

    Args:
        image_path:             Path that image location on desktop.
        output:                 Image folder that contains processed images.
    """

    current_img_name, current_img_dot_ext = os.path.splitext(image_path.split(os.path.sep)[-1])

    image = cv2.imread(image_path)

    # loop over the rotation angles again, this time ensuring
    # no part of the image is cut off
    # for angle in numpy.arange(0, 360, 10):
    for angle in [0, 10, 20, 30, 120, 130, 340, 350]:
        rotated_img = rotate_bound(image, angle)

        cv2.imwrite(f'{output}/{current_img_name}_{angle}{current_img_dot_ext}', rotated_img)

        # cv2.imshow("Rotated (Correct)", rotated_img)
        # cv2.waitKey(0


def resize_image(image_path, output):
    """Method to resizing image that in given image_path.

    Args:
        image_path:             Path that image location on desktop.
        output:                 Image folder that contains processed images.
    """

    current_img_name, current_img_dot_ext = os.path.splitext(image_path.split(os.path.sep)[-1])

    image = cv2.imread(image_path)

    half = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    bigger = cv2.resize(image, (1050, 1610))

    stretch_near = cv2.resize(image, (780, 540), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(f'{output}/{current_img_name}_orj{current_img_dot_ext}', image)
    cv2.imwrite(f'{output}/{current_img_name}_half{current_img_dot_ext}', half)
    cv2.imwrite(f'{output}/{current_img_name}_bigger{current_img_dot_ext}', bigger)
    cv2.imwrite(f'{output}/{current_img_name}_stretch_near{current_img_dot_ext}', stretch_near)


def shift_image(image_path, output):
    """Method to shifting image as horizontally or vertically that in given image_path.

    Args:
        image_path:             Path that image location on desktop.
        output:                 Image folder that contains processed images.
    """
    current_img_name, current_img_dot_ext = os.path.splitext(image_path.split(os.path.sep)[-1])

    img = load_img(image_path)
    data = img_to_array(img)
    # expand dimension to one sample
    samples = expand_dims(data, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(height_shift_range=0.15)
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for c in range(3):
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')

        cv2.imwrite(f'{output}/{current_img_name}_height_shift_{c}{current_img_dot_ext}', image)

    # create image data augmentation generator
    datagen = ImageDataGenerator(width_shift_range=[-40, 40])
    # prepare iterator
    it = datagen.flow(samples, batch_size=1)
    # generate samples and plot
    for c in range(3):
        # generate batch of images
        batch = it.next()
        # convert to unsigned integers for viewing
        image = batch[0].astype('uint8')
        cv2.imwrite(f'{output}/{current_img_name}_width_shift_{c}{current_img_dot_ext}', image)



