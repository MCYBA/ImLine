#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
.. module:: pointing
    :platform: Unix
    :synopsis: the top-level submodule of ImLine that contains the methods and classes related to ImLine's ability that is marking key points and works for creating an angular map from them.

.. moduleauthor:: Cem Baybars GÜÇLÜ <cem.baybars@gmail.com>
"""

import cv2
import os  # Miscellaneous operating system interfaces
import stasm
import csv
import numpy as np

from imutils import paths
from tinydb import Query  # TinyDB is a lightweight document oriented database
from math import sqrt, pi, cos, sin, acos, atan
from multiprocessing import Pool

from imline.db_fetching import DBFetcher
from imline.delanuay2D import Delaunay2D
from imline import IMLINE_PATH, dot_imline_dir
from imline import log_manager

logger = log_manager.get_logger(__name__, "DEBUG")


def degree_to_radian(angle):
    """The top-level method to provide converting degree type angle to radian type angle.

    Args:
        angle:       	         Servo motor's angle. Between 0 - 180 Degree.
    """

    return (angle * pi) / 180


def radian_to_degree(angle):
    """The top-level method to provide converting radian type angle to degree type angle.

    Args:
        angle:       	         Servo motor's angle. Between 0 - 3.1416(pi number) radian.
    """

    return (angle * 180) / pi


class Mapper:
    """Class to define an ability that is for mapping with lines that creates from key-points.

    This class provides necessary initiations and functions named :func:`t_system.recordation.RecordManager.start`
    for creating a Record object and start recording by this object.
    """

    def __init__(self, args):
        """Initialization method of :class:`t_system.mapping.Mapper` class.

        Args:
                args:                   Command-line arguments.
        """
        self.draw_maps = args["draw_maps"]
        self.dataset_label = args["label"]
        self.point_link_type = args["point_link_type"]

        self.out_folder = args["output"] if args["output"] else f'{dot_imline_dir}/out_{len(next(os.walk(dot_imline_dir))[1])}'
        self.mapped_imgs_folder = f'{self.out_folder}/mapped_images'
        self.environment = args["environment"]
        logger.debug(f'environment is {args["environment"]}')
        self.__check_folders()

        self.db = DBFetcher(self.out_folder, "db").fetch()

        self.current_img = None
        self.current_img_backup = None
        self.current_img_folder = None
        self.current_img_name = None
        self.current_img_dot_ext = None

        self.key_points = []
        self.current_key_point = ()

        self.valid_delanuay_triangle_count = None

        self.current_sorted_line_groups = []

    def start_by(self, raw_dataset=None, ripe_dataset=None):
        """Method to start mapping of given dataset. If raw dataset given, first start the specifying key-points.

        Args:
            raw_dataset:                Image dataset folder that contains only unprocessed images.
            ripe_dataset:               Dataset folder that contains images and key-points data.
        """

        if raw_dataset:

            image_paths = list(paths.list_images(raw_dataset))

            for (i, image_paths) in enumerate(image_paths):
                # extract the person name from the image path
                print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))

                self.current_img_name, self.current_img_dot_ext = os.path.splitext(image_paths.split(os.path.sep)[-1])

                self.current_img = cv2.imread(image_paths)
                self.current_img_backup = self.current_img.copy()

                cv2.namedWindow(self.current_img_name)
                cv2.setMouseCallback(self.current_img_name, self.__mark_key_points)

                key = None
                while True:
                    # display the image and wait for a keypress
                    cv2.imshow(self.current_img_name, self.current_img)
                    key = cv2.waitKey(1) & 0xFF

                    # if the 'r' key is pressed, reset the specified key-points.
                    if key == ord("r"):
                        self.current_img = self.current_img_backup.copy()
                        self.key_points = []

                    # if the 'o' key is pressed, record the selected key-point.
                    elif key == ord("o"):
                        if self.current_key_point:
                            if self.current_key_point in self.key_points:
                                pass
                            else:
                                self.key_points.append(self.current_key_point)
                            logger.debug(f'key points now: {self.key_points}')

                    # if the 'n' key is pressed, break from the loop and start specifying key-points of next image.
                    elif key == ord("n"):
                        self.db_upsert()
                        self.create_map()
                        self.draw_current_sorted_line_groups()
                        self.key_points = []
                        break

                    elif key == ord("f"):
                        break

                if key == ord("f"):
                    break

    def start_by_stasm(self, dataset=None):
        """Method to start mapping of given dataset. If raw dataset given, first start the specifying key-points.

        Args:
            dataset:                Image dataset folder that contains only unprocessed images.
        """
        # if not os.path.exists(dataset):
        #     os.mkdir(dataset)

        with open(f'{dataset}/map.csv', 'a', newline='') as key_point_map_dataset:
            writer = csv.writer(key_point_map_dataset)

            image_paths = list(paths.list_images(dataset))

            for (i, image_path) in enumerate(image_paths):
                # extract the person name from the image path
                print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))

                self.current_img_name, self.current_img_dot_ext = os.path.splitext(image_path.split(os.path.sep)[-1])

                self.current_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                self.current_img_backup = cv2.imread(image_path)

                landmarks = stasm.search_single(self.current_img)

                logger.debug(f'key points are {len(list(landmarks))}')

                key_points = [(landmark[0], landmark[1]) for landmark in list(landmarks)]

                if key_points:

                    # active_points = key_points[0:28] + key_points[30:31] + key_points[34:35] + key_points[40:41] + key_points[44:45] + key_points[49:50] + key_points[51:52] + key_points[53:54] + key_points[56:57] + key_points[59:60] + key_points[61:64] + key_points[65:66] + key_points[73:76]
                    # key_points[3:4] means [key_points[3]]
                    active_points = key_points[0:52] + key_points[53:]

                    self.create_map(ref_point=key_points[52], key_points=active_points)

                    intermediate_angles = []
                    for group in self.current_sorted_line_groups:
                        for intermediate_angle in group["intermediate_angles"]:
                            if intermediate_angle != 0.0:
                                intermediate_angles.append(intermediate_angle)

                    if i == 0:
                        column_labels = []
                        for (c, intermediate_angle) in enumerate(intermediate_angles):
                            column_labels.append(f'ANGLE_{c}')

                        column_labels.append(f'DATASET_LABEL')
                        writer.writerow(column_labels)

                    if intermediate_angles:
                        intermediate_angles.append(self.dataset_label)
                        writer.writerow(intermediate_angles)

                    # self.db_upsert()
                    if self.draw_maps:
                        self.draw_current_sorted_line_groups()
                        self.draw_key_points(active_points)

            self.key_points = []

    def start_by_haarcascade(self, dataset=None):
        """Method to start mapping of given dataset. If raw dataset given, first start the specifying key-points.

        Args:
            dataset:                Image dataset folder that contains only unprocessed images.
        """

        ccade_xml_file = f'{IMLINE_PATH}/haarcascade/frontalface_default.xml'
        object_cascade = cv2.CascadeClassifier(ccade_xml_file)

        with open(f'{dataset}/map.csv', 'w', newline='') as key_point_map_dataset:
            writer = csv.writer(key_point_map_dataset)

            image_paths = list(paths.list_images(dataset))

            for (i, image_path) in enumerate(image_paths):
                # extract the person name from the image path
                print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))

                self.current_img_backup = cv2.imread(image_path)
                self.current_img = cv2.cvtColor(self.current_img_backup, cv2.IMREAD_GRAYSCALE)

                # gray = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2GRAY)
                # gray = cv2.equalizeHist(gray)
                objects = object_cascade.detectMultiScale(self.current_img, 1.3, 5)

                for (x, y, w, h) in objects:
                    self.current_img_name, self.current_img_dot_ext = os.path.splitext(image_path.split(os.path.sep)[-1])

                    landmarks = stasm.search_single(self.current_img[y - 15:y + h + 15, x - 15:x + w + 15])

                    logger.debug(f'key points are {len(list(landmarks))}')

                    key_points = [(landmark[0], landmark[1]) for landmark in list(landmarks)]

                    if key_points:

                        active_points = key_points[0:28] + key_points[30:31] + key_points[34:35] + key_points[40:41] + key_points[44:45]
                        self.create_map(ref_point=key_points[52], key_points=active_points)

                        intermediate_angles = []
                        for group in self.current_sorted_line_groups:
                            for intermediate_angle in group["intermediate_angles"]:
                                if intermediate_angle != 0.0:
                                    intermediate_angles.append(intermediate_angle)

                        writer.writerow(intermediate_angles)

                        # self.db_upsert()
                        if self.draw_maps:
                            self.draw_current_sorted_line_groups()
                            self.draw_key_points(active_points)

            self.key_points = []

    def _process_stasm_image(self, writer, image_path):
        """Method to processing images by stasm requirements.

        Args:
            writer:                 CSV writer object
            image_path:             Path of Image that processing
        """

        # extract the person name from the image path
        # print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))

        self.current_img_name, self.current_img_dot_ext = os.path.splitext(image_path.split(os.path.sep)[-1])

        self.current_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.current_img_backup = cv2.imread(image_path)

        landmarks = stasm.search_single(self.current_img)

        logger.debug(f'key points are {len(list(landmarks))}')

        key_points = [(landmark[0], landmark[1]) for landmark in list(landmarks)]

        if key_points:

            # active_points = key_points[0:28] + key_points[30:31] + key_points[34:35] + key_points[40:41] + key_points[44:45] + key_points[49:50] + key_points[51:52] + key_points[53:54] + key_points[56:57] + key_points[59:60] + key_points[61:64] + key_points[65:66] + key_points[73:76]
            # key_points[3:4] means [key_points[3]]
            active_points = key_points[0:52] + key_points[53:]

            self.create_map(ref_point=key_points[52], key_points=active_points)

            intermediate_angles = []
            for group in self.current_sorted_line_groups:
                for intermediate_angle in group["intermediate_angles"]:
                    if intermediate_angle != 0.0:
                        intermediate_angles.append(intermediate_angle)

            if i == 0:
                column_labels = []
                for (c, intermediate_angle) in enumerate(intermediate_angles):
                    column_labels.append(f'ANGLE_{c}')

                column_labels.append(f'DATASET_LABEL')
                writer.writerow(column_labels)

            if intermediate_angles:
                intermediate_angles.append(self.dataset_label)
                writer.writerow(intermediate_angles)

            # self.db_upsert()
            if self.draw_maps:
                self.draw_current_sorted_line_groups()
                self.draw_key_points(active_points)

    def create_map(self, ref_point=None, key_points=None):
        """Method to create angular map with marked key points of the image.

        Args:
            ref_point:                  The common key point of the map.
            key_points:                 All key points on the image.
        """
        line_groups = []

        if self.point_link_type == "straight":
            line_groups = self.__link_points_straight(ref_point, key_points)

        elif self.point_link_type == "delanuay":
            line_groups = self.__link_points_delanuay(ref_point, key_points)

        elif self.point_link_type == "point_to_points":
            link_groups = [
                [14, 0, 12],
                [14, 53, 57],
                [49, 0, 12],
                [55, 15, 13],
                [61, 18, 25],
                [69, 0, 12],
                [73, 53, 57],
                [6, 0, 12],
                [53, 14, 73],
                [57, 14, 73],
                [0, 14, 49, 69, 6],
                [12, 14, 49, 69, 6],
            ]
            line_groups = self.__link_point_to_points(ref_point, key_points, link_groups)

        self.__sort_line_groups(line_groups)
        self.__calc_intermediate_angles()

    def __link_points_straight(self, ref_point=None, key_points=None):
        """Method to create line groups with straight linked key points.

        Args:
            ref_point:                  The common key point of the map.
            key_points:                 All key points on the image.
        """
        line_groups = []

        if key_points is not None:
            self.key_points = key_points

        if ref_point:
            last_points = []

            for point in self.key_points:

                if point != ref_point:
                    last_points.append({"point": point, "angle": self.__calc_horizontal_angle_of(ref_point, point)})

            line_groups.append({"common_point": ref_point, "last_points": last_points, "intermediate_angles": []})

        else:
            for start_point in self.key_points:

                last_points = []

                for end_point in self.key_points:
                    if start_point != end_point:
                        last_points.append({"point": end_point, "angle": self.__calc_horizontal_angle_of(start_point, end_point)})

                line_groups.append({"common_point": start_point, "last_points": last_points, "intermediate_angles": []})

        return line_groups

    def __link_point_to_points(self, ref_point, key_points, link_groups):
        """Method to create line groups with given linked group list Each list item contains 1 common point and several last points.

        Args:
            ref_point:                  The common key point of the map.
            key_points:                 All key points on the image.
            link_groups (list):         List of links. That mean its a list of list and each items keeps common point in their first item,
                                        and last points in others items. Like [[common, last1, last2], [common2, last1, last2],...]
        """
        line_groups = []

        key_points = key_points + [ref_point]

        for link_group in link_groups:

            last_points = []

            for index in link_group[1:]:
                last_points.append({"point": key_points[index], "angle": self.__calc_horizontal_angle_of(key_points[link_group[0]], key_points[index])})

            line_groups.append({"common_point": key_points[link_group[0]], "last_points": last_points, "intermediate_angles": []})

        return line_groups

    def __link_points_delanuay(self, ref_point=None, key_points=None):
        """Method to create line groups with delanuay triangular linked key points.

        Args:
            ref_point:                  The common key point of the map.
            key_points:                 All key points on the image.
        """
        line_groups = []

        key_points = key_points + [ref_point]

        dt = Delaunay2D()

        for key_point in key_points:
            try:
                dt.addPoint(key_point)
                line_groups.append({"common_point": key_point, "last_points": [], "intermediate_angles": []})
                # logger.debug(f'Delaunay triangles: {dt.exportTriangles()}')
            except Exception as e:
                logger.warning(f'Key point ignoring cause of {e}')

        triangle_count = len(dt.exportTriangles())

        if self.valid_delanuay_triangle_count:
            if self.valid_delanuay_triangle_count != triangle_count:
                return []
        else:
            self.valid_delanuay_triangle_count = triangle_count

        for s, (a, b, c) in enumerate(dt.exportTriangles()):
            logger.info(f'Triangle {s} / {triangle_count}')
            for i, line_group in enumerate(line_groups):
                # is_common_point_processed = False
                points_over_status = [False, False, False]  # List members for a, b, and c recursively.
                if line_group["last_points"]:
                    if line_group["common_point"] == key_points[a]:
                        active_points = [b, c]
                        for lg_last_point in line_group["last_points"]:
                            if lg_last_point["point"] == key_points[b]:
                                if b in active_points:
                                    active_points.remove(b)
                            elif lg_last_point["point"] == key_points[c]:
                                if c in active_points:
                                    active_points.remove(c)

                        for active_point in active_points:
                            last_point = {"point": key_points[active_point], "angle": self.__calc_horizontal_angle_of(key_points[a], key_points[active_point])}
                            line_groups[i]["last_points"].append(last_point)
                        # is_common_point_processed = True
                        points_over_status[0] = True

                    elif line_group["common_point"] == key_points[b]:
                        active_points = [a, c]
                        for lg_last_point in line_group["last_points"]:

                            if not lg_last_point["point"] == key_points[a]:
                                active_points.append(a)
                            if not lg_last_point["point"] == key_points[c]:
                                active_points.append(c)
                        for active_point in active_points:
                            last_point = {"point": key_points[active_point], "angle": self.__calc_horizontal_angle_of(key_points[b], key_points[active_point])}
                            line_groups[i]["last_points"].append(last_point)
                        # is_common_point_processed = True
                        points_over_status[1] = True

                    elif line_group["common_point"] == key_points[c]:
                        active_points = [b, a]
                        for lg_last_point in line_group["last_points"]:

                            if not lg_last_point["point"] == key_points[b]:
                                active_points.append(b)
                            if not lg_last_point["point"] == key_points[a]:
                                active_points.append(a)
                        for active_point in active_points:
                            last_point = {"point": key_points[active_point], "angle": self.__calc_horizontal_angle_of(key_points[c], key_points[active_point])}
                            line_groups[i]["last_points"].append(last_point)
                        # is_common_point_processed = True
                        points_over_status[2] = True

                else:
                    if line_group["common_point"] == key_points[a]:
                        active_points = [b, c]
                        for active_point in active_points:
                            last_point = {"point": key_points[active_point], "angle": self.__calc_horizontal_angle_of(key_points[a], key_points[active_point])}
                            line_groups[i]["last_points"].append(last_point)
                        # is_common_point_processed = True
                        points_over_status[0] = True

                    elif line_group["common_point"] == key_points[b]:
                        active_points = [a, c]
                        for active_point in active_points:
                            last_point = {"point": key_points[active_point], "angle": self.__calc_horizontal_angle_of(key_points[b], key_points[active_point])}
                            line_groups[i]["last_points"].append(last_point)
                        points_over_status[1] = True
                        # is_common_point_processed = True

                    elif line_group["common_point"] == key_points[c]:
                        active_points = [b, a]
                        for active_point in active_points:
                            last_point = {"point": key_points[active_point], "angle": self.__calc_horizontal_angle_of(key_points[c], key_points[active_point])}
                            line_groups[i]["last_points"].append(last_point)
                        # is_common_point_processed = True
                        points_over_status[2] = True

                if False not in points_over_status:
                    break

        return line_groups

    def __save_mapped_img(self):
        """Method to save image after mapping process ended.
        """
        img_folder = f'{self.mapped_imgs_folder}/{self.current_img_name}'

        if not os.path.exists(img_folder):
            os.mkdir(img_folder)

        cv2.imwrite(f'{img_folder}/_{len(next(os.walk(img_folder))[2])}{self.current_img_dot_ext}', self.current_img)

    def __get_middle_point(self):
        """Method to find and get point that middle of the image by other key points.
        """
        if self.key_points:
            img_height = self.current_img_backup.shape[0]
            img_width = self.current_img_backup.shape[1]

            middle_x = img_width / 2 + 1
            middle_y = img_height / 2 + 1

            middle_point = self.key_points[0]

            for point in self.key_points:
                if sqrt((point[0] - middle_x) ** 2 + (point[1] - middle_y) ** 2) < sqrt((middle_point[0] - middle_x) ** 2 + (middle_point[1] - middle_y) ** 2):
                    middle_point = point

            return middle_point

        return None

    @staticmethod
    def __calc_angle_between_lines(common_point, f_l_point, s_l_point):
        """Method to calculate angles that between drawed lines.

        Args:
            common_point (tuple):       The point that 2 lines crossed on it.
            f_l_point (tuple):   The point that below the first line.
            s_l_point (tuple):  The point that below the second line.
        """

        m_line_1 = (-1 * (f_l_point[1] - common_point[1])) / (f_l_point[0] - common_point[0])
        m_line_2 = (-1 * (s_l_point[1] - common_point[1])) / (s_l_point[0] - common_point[0])

        m_diff = m_line_1 - m_line_2
        m_denominator = (1 + (m_line_1 * m_line_2))

        if m_denominator == 0:
            if m_diff < 0:
                return -90
            return 90

        return round(radian_to_degree(abs(atan(m_diff / m_denominator))), 2)

    @staticmethod
    def __calc_horizontal_angle_of(first_p, last_p):
        """Method to calculate line's angle between x-axis.

        Args:
            first_p (tuple):       Line's first point.
            last_p (tuple):        Line's last point.
        """

        x = last_p[0] - first_p[0]
        y = -1 * (last_p[1] - first_p[1])

        if x == 0:
            if y < 0:
                return 270
            return 90

        angle = radian_to_degree(atan(y / x))

        if last_p[0] > first_p[0]:
            if last_p[1] > first_p[1]:
                angle += 360
        elif last_p[0] < first_p[0]:
            angle += 180

        return round(angle, 2)

    def __sort_line_groups(self, line_groups):
        """Method to sort line groups for determining current_sorted_line_group.

        Args:
                line_groups (list):      The line list that created from key-points.
        """
        self.current_sorted_line_groups = []

        for group in line_groups:
            swapped = True
            while swapped:
                swapped = False
                for i in range(len(group["last_points"]) - 1):
                    if group["last_points"][i]["angle"] > group["last_points"][i + 1]["angle"]:
                        # Swap the elements
                        group["last_points"][i]["angle"], group["last_points"][i + 1]["angle"] = group["last_points"][i + 1]["angle"], group["last_points"][i]["angle"]
                        group["last_points"][i]["point"], group["last_points"][i + 1]["point"] = group["last_points"][i + 1]["point"], group["last_points"][i]["point"]
                        # Set the flag to True so we'll loop again
                        swapped = True

            self.current_sorted_line_groups.append(group)

    def draw_current_sorted_line_groups(self):
        """Method to draw current_sorted_line_group with their intermediate angles.
        """

        for group in self.current_sorted_line_groups:
            self.current_img = self.current_img_backup
            radius = 15
            cv2.circle(self.current_img, group["common_point"], radius=radius, color=[0, 255, 0], thickness=1, lineType=8, shift=0)

            if self.environment == "testing":
                cv2.putText(self.current_img, str(group["common_point"]), group["common_point"], cv2.FONT_HERSHEY_SIMPLEX, radius * 0.01, (0, 0, 200), 1, cv2.LINE_AA)
                self.__draw_axes(self.current_img, group["common_point"])

            for (i, last_point) in enumerate(group["last_points"]):
                cv2.line(self.current_img, group["common_point"], last_point["point"], (0, 255, 0), 1, cv2.LINE_AA)

                intermediate_angle = group["intermediate_angles"][i]
                if group["intermediate_angles"][i] == 0:
                    intermediate_angle = 0.0001

                angle_x = int(group["common_point"][0] + radius * 1.5 * sqrt(sqrt(360 / intermediate_angle)) * cos(degree_to_radian(last_point["angle"] + (intermediate_angle / 2))))
                angle_y = int(group["common_point"][1] - radius * 1.5 * sqrt(sqrt(360 / intermediate_angle)) * sin(degree_to_radian(last_point["angle"] + (intermediate_angle / 2))))

                image_height, image_width = self.current_img.shape[:2]

                if angle_x >= image_width - 15:
                    angle_x = image_width - 15
                elif angle_x <= 0:
                    angle_x = 15

                if angle_y >= image_height:
                    angle_y = image_height - 15
                elif angle_y <= 0:
                    angle_y = 15

                cv2.putText(self.current_img, str(group["intermediate_angles"][i]), (angle_x, angle_y), cv2.FONT_HERSHEY_SIMPLEX, radius * 0.009, (0, 0, 200), 1, cv2.LINE_AA)

            self.__save_mapped_img()

    def draw_key_points(self, key_points):
        """Method to draw key points that belongs to the given face image.

        Args:
                key_points:              Points that becomes by physical structure of face.
        """
        self.current_img = self.current_img_backup.copy()

        for i, point in enumerate(key_points):
            # cv2.rectangle(self.current_img, point, point, (0, 255, 0), 2)
            cv2.putText(self.current_img, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 255, 0), 1, cv2.LINE_AA)

        self.__save_mapped_img()

    @staticmethod
    def __draw_axes(img, point):
        """Method to draw coordinate axes by origin point.

        Args:
            img                         Image that will recorded.
            point:                      Origin of the axes.
        """
        radius = 75

        cv2.line(img, point, (point[0] + radius * 3, point[1]), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, point, (point[0] - radius * 3, point[1]), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, point, (point[0], point[1] + radius * 3), (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(img, point, (point[0], point[1] - radius * 3), (255, 0, 0), 1, cv2.LINE_AA)

    def __calc_intermediate_angles(self):
        """Method to calculate angle between all lines of the group.
        """

        for group in self.current_sorted_line_groups:
            total_angle = 0
            for (i, point) in enumerate(group["last_points"]):

                if i == len(group["last_points"]) - 1:
                    angle = 360 - total_angle
                else:
                    angle = group["last_points"][i + 1]["angle"] - point["angle"]
                    total_angle += angle

                group["intermediate_angles"].append(round(angle, 2))

    def __mark_key_points(self, event, x, y, flags, param):
        """Method to marking key-points on images in given dataset.
        """

        if event == cv2.EVENT_LBUTTONDOWN:

            if self.current_key_point:
                if self.current_key_point in self.key_points:
                    pass
                else:
                    self.current_img[self.current_key_point[1], self.current_key_point[0]] = self.current_img_backup[self.current_key_point[1], self.current_key_point[0]]
                    cv2.imshow(self.current_img_name, self.current_img)

            self.current_key_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if not (x, y) == self.current_key_point:
                self.current_key_point = ()
            else:
                cv2.rectangle(self.current_img, self.current_key_point, self.current_key_point, (0, 255, 0), 2)
                cv2.imshow(self.current_img_name, self.current_img)

    def db_upsert(self, force_insert=False):
        """Function to insert(or update) the position to the database.

        Args:
            force_insert (bool):    Force insert flag.

        Returns:
            str:  Response.
        """

        if self.db.search((Query().name == self.current_img_name)):
            if force_insert:
                # self.already_exist = False
                self.db.update({'key_points': self.key_points, 'line_groups': self.current_sorted_line_groups}, Query().name == self.current_img_name)

            else:
                # self.already_exist = True
                return "Already Exist"
        else:
            self.db.insert({
                'name': self.current_img_name,
                'key_points': self.key_points,
                'line_groups': self.current_sorted_line_groups
            })  # insert the given data

        return ""

    def __check_folders(self):
        """Method to checking the necessary folders created before. If not created creates them.
        """

        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)

        if not os.path.exists(self.mapped_imgs_folder):
            os.mkdir(self.mapped_imgs_folder)
