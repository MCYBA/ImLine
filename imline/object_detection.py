#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. module:: pointing
    :platform: Unix
    :synopsis: the top-level submodule of ImLine that contains the methods and classes related to ImLine's ability that is marking face location in dataset images.

.. moduleauthor:: Cem Baybars GÜÇLÜ <cem.baybars@gmail.com>
"""

import cv2
import os  # Miscellaneous operating system interfaces
import numpy

from imutils import paths
from math import sqrt, radians, cos, sin

from imline import IMLINE_PATH, dot_imline_dir
from imline import log_manager

logger = log_manager.get_logger(__name__, "DEBUG")


def mark_object(dataset, output):
    """Method to marking face locations in images of given dataset.

    Args:
        dataset:                Image dataset folder that contains only unprocessed images.
        output:                 Image folder that contains processed images.
    """

    ccade_xml_file = f'{IMLINE_PATH}/haarcascade/frontalface_default.xml'
    object_cascade = cv2.CascadeClassifier(ccade_xml_file)

    image_paths = list(paths.list_images(dataset))

    for (i, image_path) in enumerate(image_paths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))

        current_img = cv2.imread(image_path)
        grey_image = cv2.cvtColor(current_img, cv2.IMREAD_GRAYSCALE)
        current_img_backup = current_img.copy()

        # gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        objects = object_cascade.detectMultiScale(grey_image, 1.3, 5)

        for (x, y, w, h) in objects:
            current_img_name, current_img_dot_ext = os.path.splitext(image_path.split(os.path.sep)[-1])
            radius = int(sqrt(w * w + h * h) / 2)
            process_image = __draw_object(current_img, (int(x + w / 2), int(y + h / 2)), radius)

            __save_mapped_img(f'{output}', process_image, current_img_name, current_img_dot_ext)


def __draw_object(image, center, radius, color='red'):
    """The top-level method to real time object detection.
    Args:
            image:       	    Image matrix.
            center:       	    Center point of the aimed object.
            radius:             Radius of the aim.
    """
    process_image = image.copy()
    rect_diagonal_rate = 0.9
    radius *= 0.5
    thickness = radius * 0.23

    image_height, image_width = image.shape[:2]
    center_x = center[0]
    center_y = center[1]
    text_font = cv2.FONT_HERSHEY_SIMPLEX

    __draw_rect(process_image, center_x, center_y, radius * 1.2, thickness * 0.4)
    # self.draw_rect_triangler(center_x, center_y, radius * self.rect_diagonal_rate, thickness * 0.2)
    text = "YUZ"

    text_point = (int(center_x + radius * 0.65), int(center_y + radius * 1.6))
    # text_point = (int(center_x - radius * 0.7), int(center_y - radius * 0.95 - radius * 0.1))
    # text_point1 = (int(center_x - radius * 0.1), int(center_y + radius * 0.95 - radius * 0.1))

    cv2.putText(process_image, text, text_point, text_font, radius * 0.008, (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.line(image, text_point, text_point1, (0, 0, 200), int(radius * 0.05), cv2.LINE_AA)
    # parameters: image, put text, text's coordinates,font, scale, color, thickness, line type(this type is best for texts.)

    # cv2.line(process_image, (int(center_x - radius * 0.95 * sin(radians(15))), int(center_y + radius * 0.95 * cos(radians(45)))), (int(center_x + radius * 0.95 * sin(radians(15))), int(center_y + radius * 0.95 * cos(radians(45)))), (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.line(process_image, (int(center_x + radius * 0.95 * sin(radians(15))), int(center_y + radius * 0.95 * cos(radians(45)))), (int(center_x + radius * 0.95 * sin(radians(20))), int(center_y + radius * 0.95 * cos(radians(15)))), (0, 255, 0), 1, cv2.LINE_AA)
    # cv2.line(process_image, (int(center_x - radius * 0.95 * sin(radians(15))), int(center_y + radius * 0.95 * cos(radians(45)))), (int(center_x - radius * 0.95 * sin(radians(20))), int(center_y + radius * 0.95 * cos(radians(15)))), (0, 255, 0), 1, cv2.LINE_AA)

    rect_diagonal_rate -= 0.05

    if rect_diagonal_rate <= 0.2:
        rect_diagonal_rate = 0.9
    return process_image


def __draw_rect(image, center_x, center_y, radius, thickness):

    image_height, image_width = image.shape[:2]

    center_x = int(center_x)
    center_y = int(center_y)
    radius = int(radius)
    thickness = int(thickness)

    # top_left_x = center_x - radius - thickness
    # top_right_x = center_x + radius + thickness
    #
    # top_left_y = center_y - radius - thickness
    # bottom_left_y = center_y + radius + thickness
    edge_length = int(radius * 0.3)
    x_ranges = list(range(center_x - radius - thickness, center_x - edge_length)) + list(range(center_x + edge_length, center_x + radius + thickness))
    y_ranges = list(range(center_y - radius - thickness, center_y - radius)) + list(range(center_y + radius, center_y + radius + thickness))

    for x in x_ranges:
        for y in y_ranges:
            if image_width > x >= 0 and image_height > y >= 0:  # for the frames' limit protection.
                [b, g, r] = image[y, x] = numpy.array(image[y, x]) * numpy.array([0, 1, 0])
                if g <= 100:
                    if g == 0:
                        g = 1
                        image[y, x] = [0, 0, 1]
                    greenness_rate = (255 / g) / 0.12
                    image[y, x] = numpy.array(image[y, x]) * numpy.array([0, greenness_rate, 0])

    y_ranges = list(range(center_y - radius - thickness, center_y - edge_length)) + list(range(center_y + edge_length, center_y + radius + thickness))
    x_ranges = list(range(center_x - radius - thickness, center_x - radius)) + list(range(center_x + radius, center_x + radius + thickness))

    for y in y_ranges:
        for x in x_ranges:
            if image_width > x >= 0 and image_height > y >= 0:  # for the frames' limit protection.
                [b, g, r] = image[y, x] = numpy.array(image[y, x]) * numpy.array([0, 1, 0])
                if g <= 100:
                    if g == 0:
                        g = 1
                        image[y, x] = [0, 0, 1]
                    greenness_rate = (255 / g) / 0.12
                    image[y, x] = numpy.array(image[y, x]) * numpy.array([0, greenness_rate, 0])

    x_ranges = list(range(int(center_x - radius * 1.5), int(center_x - edge_length))) + list(range(int(center_x + edge_length), int(center_x + radius * 1.5)))

    for x in x_ranges:
        if image_width > x >= 0:  # for the frames' limit protection.
            image[center_y, x] = numpy.array(image[center_y, x]) * numpy.array([0, 2, 0])

    y_ranges = list(range(int(center_y - radius * 1.5), int(center_y - edge_length))) + list(range(int(center_y + edge_length), int(center_y + radius * 1.5)))

    for y in y_ranges:
        if image_height > y >= 0:  # for the frames' limit protection.
            image[y, center_x] = numpy.array(image[y, center_x]) * numpy.array([0, 2, 0])

    # [b, g, r] = image[y, x] = numpy.array(image[y, x]) * numpy.array([1.2, 1.2, 0])
    # if b <= 100 or g <= 100:
    #     if b == 0:
    #         b = 1
    #         image[y, x] = [1, 0, 0]
    #     if g == 0:
    #         g = 1
    #         image[y, x] = [0, 1, 0]
    #     blueness_rate = (255 / b) / 0.12
    #     greenness_rate = (255 / g) / 0.12
    #     image[y, x] = numpy.array(image[y, x]) * numpy.array([blueness_rate, greenness_rate, 0])


def __draw_rect_triangler(image, center_x, center_y, radius, thickness):
    center_x = int(center_x)
    center_y = int(center_y)
    radius = int(radius)
    thickness = int(thickness)
    edge_length = int(radius * 0.3)

    for x in range(center_x - radius - thickness, center_x - edge_length):
        edge_y = radius + thickness - edge_length - (x - center_x + radius + thickness)

        for y in range(center_y - radius - thickness, edge_y + center_y - radius - thickness):
            image[y, x] = numpy.array(image[y, x]) * numpy.array([1.2, 1.2, 0])

        for y in range(center_y + radius + thickness - edge_y, center_y + radius + thickness):
            image[y, x] = numpy.array(image[y, x]) * numpy.array([1.2, 1.2, 0])

    for x in range(center_x + edge_length, center_x + radius + thickness):
        edge_y = (x - center_x - edge_length)

        for y in range(center_y - radius - thickness, edge_y + center_y - radius - thickness):
            image[y, x] = numpy.array(image[y, x]) * numpy.array([1.2, 1.2, 0])

        for y in range(center_y + radius + thickness - edge_y, center_y + radius + thickness):
            image[y, x] = numpy.array(image[y, x]) * numpy.array([1.2, 1.2, 0])


def __save_mapped_img(folder, image, img_name, img_dot_ext):
    """Method to save image after mapping process ended.
    """
    img_folder = f'{folder}/{img_name}'

    if not os.path.exists(img_folder):
        os.mkdir(img_folder)

    cv2.imwrite(f'{img_folder}/_{len(next(os.walk(img_folder))[2])}{img_dot_ext}', image)

