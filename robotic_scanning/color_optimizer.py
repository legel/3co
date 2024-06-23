#!/usr/bin/env python
"""Python-Macduff: "the Macbeth ColorChecker finder", ported to Python.

Original C++ code: github.com/ryanfb/macduff/

Usage:
    # if pixel-width of color patches is unknown,
    $ python macduff.py examples/test.jpg result.png > result.csv

    # if pixel-width of color patches is known to be, e.g. 65,
    $ python macduff.py examples/test.jpg result.png 65 > result.csv
"""
from __future__ import print_function, division
import cv2 as cv
import numpy as np
from numpy.linalg import norm
from math import sqrt
from sys import stderr, argv
from copy import copy
import os
from scipy.optimize import minimize, brute, fmin
import matplotlib.pyplot as plt
import matplotlib.image as image
from PIL import Image, ImageCms
import random
from robotics import Iris
import sys

robot = Iris()
robot.scan(project_name="color_field_z_0cm_pitch_-90degrees_exposure_x", export_to_npy=True, process_point_cloud=True, auto_focus=True, auto_exposure=True, save_16_bit_image=False)

sys.exit(0)

iteration_number = 0

_root = os.path.dirname(os.path.realpath(__file__))


# Each color square must takes up more than this percentage of the image
MIN_RELATIVE_SQUARE_SIZE = 0.0001

DEBUG = False

MACBETH_WIDTH = 6
MACBETH_HEIGHT = 4
MACBETH_SQUARES = MACBETH_WIDTH * MACBETH_HEIGHT

MAX_CONTOUR_APPROX = 50  # default was 7


# pick the colorchecker values to use -- several options available in
# the `color_data` subdirectory
# Note: all options are explained in detail at
# http://www.babelcolor.com/colorchecker-2.htm
color_data = os.path.join(_root, 'color_data',
                          'xrite_passport_colors_sRGB-GMB-2005.csv')
expected_colors = np.flip(np.loadtxt(color_data, delimiter=','), 1)
expected_colors = expected_colors.reshape(MACBETH_HEIGHT, MACBETH_WIDTH, 3)


# a class to simplify the translation from c++
class Box2D:
    """
    Note: The Python equivalent of `RotatedRect` and `Box2D` objects 
    are tuples, `((center_x, center_y), (w, h), rotation)`.
    Example:
    >>> cv.boxPoints(((0, 0), (2, 1), 0))
    array([[-1. ,  0.5],
           [-1. , -0.5],
           [ 1. , -0.5],
           [ 1. ,  0.5]], dtype=float32)
    >>> cv.boxPoints(((0, 0), (2, 1), 90))
    array([[-0.5, -1. ],
           [ 0.5, -1. ],
           [ 0.5,  1. ],
           [-0.5,  1. ]], dtype=float32)
    """

    def __init__(self, center=None, size=None, angle=0, rrect=None):
        if rrect is not None:
            center, size, angle = rrect

        # self.center = Point2D(*center)
        # self.size = Size(*size)
        self.center = center
        self.size = size
        self.angle = angle  # in degrees

    def rrect(self):
        return self.center, self.size, self.angle


def crop_patch(center, size, image):
    """Returns mean color in intersection of `image` and `rectangle`."""
    x, y = center - np.array(size)/2
    w, h = size
    x0, y0, x1, y1 = map(round, [x, y, x + w, y + h])
    return image[int(max(y0, 0)): int(min(y1, image.shape[0])),
                 int(max(x0, 0)): int(min(x1, image.shape[1]))]


def contour_average(contour, image):
    """Assuming `contour` is a polygon, returns the mean color inside it.

    Note: This function is inefficiently implemented!!! 
    Maybe using drawing/fill functions would improve speed.
    """

    # find up-right bounding box
    xbb, ybb, wbb, hbb = cv.boundingRect(contour)

    # now found which points in bounding box are inside contour and sum
    def is_inside_contour(pt):
        return cv.pointPolygonTest(contour, pt, False) > 0

    from itertools import product as catesian_product
    from operator import add
    from functools import reduce
    bb = catesian_product(range(max(xbb, 0), min(xbb + wbb,  image.shape[1])),
                          range(max(ybb, 0), min(ybb + hbb,  image.shape[0])))
    pts_inside_of_contour = [xy for xy in bb if is_inside_contour(xy)]

    # pts_inside_of_contour = list(filter(is_inside_contour, bb))
    color_sum = reduce(add, (image[y, x] for x, y in pts_inside_of_contour))
    return color_sum / len(pts_inside_of_contour)


def rotate_box(box_corners):
    """NumPy equivalent of `[arr[i-1] for i in range(len(arr)]`"""
    return np.roll(box_corners, 1, 0)


def check_colorchecker(values, expected_values=expected_colors):
    """Find deviation of colorchecker `values` from expected values."""
    diff = (values - expected_values).ravel(order='K')
    return sqrt(np.dot(diff, diff))


def draw_colorchecker(colors, centers, image, radius):
    for observed_color, expected_color, pt in zip(colors.reshape(-1, 3),
                                                  expected_colors.reshape(-1, 3),
                                                  centers.reshape(-1, 2)):
        x, y = pt
        cv.circle(image, (round(x), round(y)), radius //
                  2, expected_color.tolist(), -1)
        cv.circle(image, (round(x), round(y)), radius //
                  4, observed_color.tolist(), -1)
    return image


class ColorChecker:
    def __init__(self, error, values, points, size):
        self.error = error
        self.values = values
        self.points = points
        self.size = size


def find_colorchecker(boxes, image, debug_filename=None, use_patch_std=True,
                      debug=DEBUG):

    points = np.array([[box.center[0], box.center[1]] for box in boxes])
    passport_box = cv.minAreaRect(points.astype('float32'))
    (x, y), (w, h), a = passport_box
    box_corners = cv.boxPoints(passport_box)

    # sort `box_corners` to be in order tl, tr, br, bl
    top_corners = sorted(enumerate(box_corners), key=lambda c: c[1][1])[:2]
    top_left_idx = min(top_corners, key=lambda c: c[1][0])[0]
    box_corners = np.roll(box_corners, -top_left_idx, 0)
    tl, tr, br, bl = box_corners

    if debug:
        debug_images = [copy(image), copy(image)]
        for box in boxes:
            pts_ = [cv.boxPoints(box.rrect()).astype(np.int32)]
            cv.polylines(debug_images[0], pts_, True, (255, 0, 0))
        pts_ = [box_corners.astype(np.int32)]
        cv.polylines(debug_images[0], pts_, True, (0, 0, 255))

        bgrp = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)]
        for pt, c in zip(box_corners, bgrp):
            cv.circle(debug_images[0], tuple(np.array(pt, dtype='int')), 10, c)
        # cv.imwrite(debug_filename, np.vstack(debug_images))

        print("Box:\n\tCenter: %f,%f\n\tSize: %f,%f\n\tAngle: %f\n"
              "" % (x, y, w, h, a), file=stderr)

    landscape_orientation = True  # `passport_box` is wider than tall
    if norm(tr - tl) < norm(bl - tl):
        landscape_orientation = False

    average_size = int(sum(min(box.size) for box in boxes) / len(boxes))
    if landscape_orientation:
        dx = (tr - tl)/(MACBETH_WIDTH - 1)
        dy = (bl - tl)/(MACBETH_HEIGHT - 1)
    else:
        dx = (bl - tl)/(MACBETH_WIDTH - 1)
        dy = (tr - tl)/(MACBETH_HEIGHT - 1)

    # calculate the averages for our oriented colorchecker
    checker_dims = (MACBETH_HEIGHT, MACBETH_WIDTH)
    patch_values = np.empty(checker_dims + (3,), dtype='float32')
    patch_points = np.empty(checker_dims + (2,), dtype='float32')
    sum_of_patch_stds = np.array((0.0, 0.0, 0.0))
    for x in range(MACBETH_WIDTH):
        for y in range(MACBETH_HEIGHT):
            center = tl + x*dx + y*dy

            px, py = center
            img_patch = crop_patch(center, [average_size]*2, image)

            if not landscape_orientation:
                y = MACBETH_HEIGHT - 1 - y

            patch_points[y, x] = center
            patch_values[y, x] = img_patch.mean(axis=(0, 1))
            sum_of_patch_stds += img_patch.std(axis=(0, 1))

            if debug:
                rect = (px, py), (average_size, average_size), 0
                pts_ = [cv.boxPoints(rect).astype(np.int32)]
                cv.polylines(debug_images[1], pts_, True, (0, 255, 0))
    if debug:
        cv.imwrite(debug_filename, np.vstack(debug_images))

    # determine which orientation has lower error
    orient_1_error = check_colorchecker(patch_values)
    orient_2_error = check_colorchecker(patch_values[::-1, ::-1])

    if orient_1_error > orient_2_error:  # rotate by 180 degrees
        patch_values = patch_values[::-1, ::-1]
        patch_points = patch_points[::-1, ::-1]

    if use_patch_std:
        error = sum_of_patch_stds.mean() / MACBETH_SQUARES
    else:
        error = min(orient_1_error, orient_2_error)

    if debug:
        print("dx =", dx, file=stderr)
        print("dy =", dy, file=stderr)
        print("Average contained rect size is %d\n" %
              average_size, file=stderr)
        print("Orientation 1: %f\n" % orient_1_error, file=stderr)
        print("Orientation 2: %f\n" % orient_2_error, file=stderr)
        print("Error: %f\n" % error, file=stderr)

    return ColorChecker(error=error,
                        values=patch_values,
                        points=patch_points,
                        size=average_size)


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))


# https://github.com/opencv/opencv/blob/master/samples/python/squares.py
# Note: This is similar to find_quads, added to hastily add support to
# the `patch_size` parameter
def find_squares(img):
    img = cv.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv.split(img):
        for thrs in range(0, 255, 26):
            if thrs == 0:
                bin = cv.Canny(gray, 0, 50, apertureSize=5)
                bin = cv.dilate(bin, None)
            else:
                _retval, bin = cv.threshold(gray, thrs, 255, cv.THRESH_BINARY)

            tmp = cv.findContours(bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            try:
                contours, _ = tmp
            except ValueError:  # OpenCV version < 4.0.0
                bin, contours, _ = tmp

            for cnt in contours:
                cnt_len = cv.arcLength(cnt, True)
                cnt = cv.approxPolyDP(cnt, 0.02*cnt_len, True)
                if (len(cnt) == 4 and cv.contourArea(cnt) > 1000
                        and cv.isContourConvex(cnt)):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i + 2) % 4])
                                   for i in range(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares


def is_right_size(quad, patch_size, rtol=.25):
    """Determines if a (4-point) contour is approximately the right size."""
    cw = abs(np.linalg.norm(quad[0] - quad[1]) - patch_size) < rtol*patch_size
    ch = abs(np.linalg.norm(quad[0] - quad[3]) - patch_size) < rtol*patch_size
    return cw and ch


# stolen from icvGenerateQuads
def find_quad(src_contour, min_size, debug_image=None):

    for max_error in range(2, MAX_CONTOUR_APPROX + 1):
        dst_contour = cv.approxPolyDP(src_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break

        # we call this again on its own output, because sometimes
        # cvApproxPoly() does not simplify as much as it should.
        dst_contour = cv.approxPolyDP(dst_contour, max_error, closed=True)
        if len(dst_contour) == 4:
            break

    # reject non-quadrangles
    is_acceptable_quad = False
    is_quad = False
    if len(dst_contour) == 4 and cv.isContourConvex(dst_contour):
        is_quad = True
        perimeter = cv.arcLength(dst_contour, closed=True)
        area = cv.contourArea(dst_contour, oriented=False)

        d1 = np.linalg.norm(dst_contour[0] - dst_contour[2])
        d2 = np.linalg.norm(dst_contour[1] - dst_contour[3])
        d3 = np.linalg.norm(dst_contour[0] - dst_contour[1])
        d4 = np.linalg.norm(dst_contour[1] - dst_contour[2])

        # philipg.  Only accept those quadrangles which are more square
        # than rectangular and which are big enough
        cond = (d3/1.1 < d4 < d3*1.1 and
                d3*d4/1.5 < area and
                min_size < area and
                d1 >= 0.15*perimeter and
                d2 >= 0.15*perimeter)

        if not cv.CALIB_CB_FILTER_QUADS or area > min_size and cond:
            is_acceptable_quad = True
            # return dst_contour
    if debug_image is not None:
        cv.drawContours(debug_image, [src_contour], -1, (255, 0, 0), 1)
        if is_acceptable_quad:
            cv.drawContours(debug_image, [dst_contour], -1, (0, 255, 0), 1)
        elif is_quad:
            cv.drawContours(debug_image, [dst_contour], -1, (0, 0, 255), 1)
        return debug_image

    if is_acceptable_quad:
        return dst_contour
    return None


def find_macbeth(img, patch_size=None, is_passport=False, debug=DEBUG,
                 min_relative_square_size=MIN_RELATIVE_SQUARE_SIZE):
    macbeth_img = img
    if isinstance(img, str):
        macbeth_img = cv.imread(img)
    macbeth_original = copy(macbeth_img)
    macbeth_split = cv.split(macbeth_img)

    # threshold each channel and OR results together
    block_size = int(min(macbeth_img.shape[:2]) * 0.02) | 1
    macbeth_split_thresh = []
    for channel in macbeth_split:
        res = cv.adaptiveThreshold(channel,
                                   255,
                                   cv.ADAPTIVE_THRESH_MEAN_C,
                                   cv.THRESH_BINARY_INV,
                                   block_size,
                                   C=6)
        macbeth_split_thresh.append(res)
    adaptive = np.bitwise_or(*macbeth_split_thresh)

    if debug:
        print("Used %d as block size\n" % block_size, file=stderr)
        cv.imwrite('debug_threshold.png',
                   np.vstack(macbeth_split_thresh + [adaptive]))

    # do an opening on the threshold image
    element_size = int(2 + block_size / 10)
    shape, ksize = cv.MORPH_RECT, (element_size, element_size)
    element = cv.getStructuringElement(shape, ksize)
    adaptive = cv.morphologyEx(adaptive, cv.MORPH_OPEN, element)

    if debug:
        print("Used %d as element size\n" % element_size, file=stderr)
        cv.imwrite('debug_adaptive-open.png', adaptive)

    # find contours in the threshold image
    tmp = cv.findContours(image=adaptive,
                          mode=cv.RETR_LIST,
                          method=cv.CHAIN_APPROX_SIMPLE)
    try:
        contours, _ = tmp
    except ValueError:  # OpenCV < 4.0.0
        adaptive, contours, _ = tmp

    if debug:
        show_contours = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
        cv.drawContours(show_contours, contours, -1, (0, 255, 0))
        cv.imwrite('debug_all_contours.png', show_contours)

    min_size = np.product(macbeth_img.shape[:2]) * min_relative_square_size

    def is_seq_hole(c):
        return cv.contourArea(c, oriented=True) > 0

    def is_big_enough(contour):
        _, (w, h), _ = cv.minAreaRect(contour)
        return w * h >= min_size

    # filter out contours that are too small or clockwise
    contours = [c for c in contours if is_big_enough(c) and is_seq_hole(c)]

    if debug:
        show_contours = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
        cv.drawContours(show_contours, contours, -1, (0, 255, 0))
        cv.imwrite('debug_big_contours.png', show_contours)

        debug_img = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
        for c in contours:
            debug_img = find_quad(c, min_size, debug_image=debug_img)
        cv.imwrite("debug_quads.png", debug_img)

    if contours:
        if patch_size is None:
            initial_quads = [find_quad(c, min_size) for c in contours]
        else:
            initial_quads = [s for s in find_squares(macbeth_original)
                             if is_right_size(s, patch_size)]
            if is_passport and len(initial_quads) <= MACBETH_SQUARES:
                qs = [find_quad(c, min_size) for c in contours]
                qs = [x for x in qs if x is not None]
                initial_quads = [x for x in qs if is_right_size(x, patch_size)]
        initial_quads = [q for q in initial_quads if q is not None]
        initial_boxes = [Box2D(rrect=cv.minAreaRect(q)) for q in initial_quads]

        if debug:
            show_quads = cv.cvtColor(copy(adaptive), cv.COLOR_GRAY2BGR)
            cv.drawContours(show_quads, initial_quads, -1, (0, 255, 0))
            cv.imwrite('debug_quads2.png', show_quads)
            print("%d initial quads found", len(initial_quads), file=stderr)

        if is_passport or (len(initial_quads) > MACBETH_SQUARES):
            if debug:
                print(" (probably a Passport)\n", file=stderr)

            # set up the points sequence for cvKMeans2, using the box centers
            points = np.array([box.center for box in initial_boxes],
                              dtype='float32')

            # partition into two clusters: passport and colorchecker
            criteria = \
                (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            compactness, clusters, centers = \
                cv.kmeans(data=points,
                          K=2,
                          bestLabels=None,
                          criteria=criteria,
                          attempts=100,
                          flags=cv.KMEANS_RANDOM_CENTERS)

            partitioned_quads = [[], []]
            partitioned_boxes = [[], []]
            for i, cluster in enumerate(clusters.ravel()):
                partitioned_quads[cluster].append(initial_quads[i])
                partitioned_boxes[cluster].append(initial_boxes[i])

            debug_fns = [None, None]
            if debug:
                debug_fns = ['debug_passport_box_%s.jpg' % i for i in (0, 1)]

                # show clustering
                img_clusters = []
                for cl in partitioned_quads:
                    img_copy = copy(macbeth_original)
                    cv.drawContours(img_copy, cl, -1, (255, 0, 0))
                    img_clusters.append(img_copy)
                cv.imwrite('debug_clusters.jpg', np.vstack(img_clusters))

            # check each of the two partitioned sets for the best colorchecker
            partitioned_checkers = []
            for cluster_boxes, fn in zip(partitioned_boxes, debug_fns):
                partitioned_checkers.append(
                    find_colorchecker(cluster_boxes, macbeth_original, fn,
                                      debug=debug))

            # use the colorchecker with the lowest error
            found_colorchecker = min(partitioned_checkers,
                                     key=lambda checker: checker.error)

        else:  # just one colorchecker to test
            debug_img = None
            if debug:
                debug_img = "debug_passport_box.jpg"
                print("\n", file=stderr)

            found_colorchecker = \
                find_colorchecker(initial_boxes, macbeth_original, debug_img,
                                  debug=debug)

        # render the found colorchecker
        draw_colorchecker(found_colorchecker.values,
                          found_colorchecker.points,
                          macbeth_img,
                          found_colorchecker.size)

        # print out the colorchecker info
        for color, pt in zip(found_colorchecker.values.reshape(-1, 3),
                             found_colorchecker.points.reshape(-1, 2)):
            b, g, r = color
            x, y = pt
            if debug:
                print("%.0f,%.0f,%.0f,%.0f,%.0f\n" % (x, y, r, g, b))
        if debug:
            print("%0.f\n%f\n"
                  "" % (found_colorchecker.size, found_colorchecker.error))
    else:
        raise Exception('Something went wrong -- no contours found')
    return macbeth_img, found_colorchecker


def calculate_edges(colorchecker):
    nested_point_list = colorchecker.points
    size = colorchecker.size
    safety_measure = 15
    edge_distance = round((size / 2) - safety_measure)

    corners_box = []

    for group in nested_point_list:
        for point in group:
            (x, y) = (round(point[0]), round(point[1]))
            x_min = x - edge_distance
            x_max = x + edge_distance
            y_min = y - edge_distance
            y_max = y + edge_distance
            corners_box.append((x_min, x_max, y_min, y_max))

    return corners_box


def colorchecker_truecolors():
    xrite_target_colors = [
        {'id': 1, 'rgb': (115, 82, 68), 'lab': (
            37.986, 13.555, 14.059), 'name': 'dark skin'},
        {'id': 2, 'rgb': (194, 150, 130), 'lab': (
            65.711, 18.130, 17.810), 'name': 'light skin'},
        {'id': 3, 'rgb': (98, 122, 157), 'lab': (
            49.927, -4.880, -21.925), 'name': 'blue sky'},
        {'id': 4, 'rgb': (87, 108, 67), 'lab': (
            43.139, -13.095, 21.905), 'name': 'foliage'},
        {'id': 5, 'rgb': (133, 128, 177), 'lab': (
            55.112, 8.844, -25.399), 'name': 'blue flower'},
        {'id': 6, 'rgb': (103, 189, 170), 'lab': (
            70.719, -33.397, -0.199), 'name': 'bluish green'},
        {'id': 7, 'rgb': (214, 126, 44), 'lab': (
            62.661, 36.067, 57.096), 'name': 'orange'},
        {'id': 8, 'rgb': (80, 91, 166), 'lab': (
            40.020, 10.410, -45.964), 'name': 'purplish blue'},
        {'id': 9, 'rgb': (193, 90, 99), 'lab': (
            51.124, 48.239, 16.248), 'name': 'moderate red'},
        {'id': 10, 'rgb': (94, 60, 108), 'lab': (
            30.325, 22.976, -21.587), 'name': 'purple'},
        {'id': 11, 'rgb': (157, 188, 64), 'lab': (
            72.532, -23.709, 57.255), 'name': 'yellow green'},
        {'id': 12, 'rgb': (224, 163, 46), 'lab': (
            71.941, 19.363, 67.857), 'name': 'orange yellow'},
        {'id': 13, 'rgb': (56, 61, 150), 'lab': (
            28.778, 14.179, -50.297), 'name': 'blue'},
        {'id': 14, 'rgb': (70, 148, 73), 'lab': (
            55.261, -38.342, 31.370), 'name': 'green'},
        {'id': 15, 'rgb': (175, 54, 60), 'lab': (
            42.101, 53.378, 28.190), 'name': 'red'},
        {'id': 16, 'rgb': (231, 199, 31), 'lab': (
            81.733, 4.039, 79.819), 'name': 'yellow'},
        {'id': 17, 'rgb': (187, 86, 149), 'lab': (
            51.935, 49.986, -14.574), 'name': 'magenta'},
        {'id': 18, 'rgb': (8, 133, 161), 'lab': (
            51.038, -28.631, -28.638), 'name': 'cyan'},
        {'id': 19, 'rgb': (243, 243, 242), 'lab': (
            96.539, -0.425, 1.186), 'name': 'white (.05 *)'},
        {'id': 20, 'rgb': (200, 200, 200), 'lab': (
            81.257, -0.638, -0.335), 'name': 'neutral 8 (.23 *)'},
        {'id': 21, 'rgb': (160, 160, 160), 'lab': (
            66.766, -0.734, -0.504), 'name': 'neutral 6.5 (.44 *)'},
        {'id': 22, 'rgb': (122, 122, 121), 'lab': (
            50.867, -0.153, -0.270), 'name': 'neutral 5 (.70 *)'},
        {'id': 23, 'rgb': (85, 85, 85), 'lab': (
            35.656, -0.421, -1.231), 'name': 'neutral 3.5 (1.05 *)'},
        {'id': 24, 'rgb': (52, 52, 52), 'lab': (
            20.461, -0.079, -0.973), 'name': 'black (1.50 *)'}
    ]
    return xrite_target_colors


def multiply_rgb_image(rgb_image, multipliers):

    def apply_multipliers(rgbvalues, multipliers):
        r, g, b = rgbvalues[0], rgbvalues[1], rgbvalues[2]
        totalrgb = r + g + b
        m_1, m_2, m_3 = multipliers[0], multipliers[1], multipliers[2]
        return r * m_1, g * m_2, b * m_3

    im = Image.open(rgb_image)
    shape = im.size
    pixel_values = np.array(im.getdata())
    multiplied_image = np.apply_along_axis(apply_multipliers, 1, pixel_values, multipliers)
    return shape, multiplied_image

def make_path_to_image(rgb_array, shape, iteration):
    (w, h) = shape
    path_to_image = 'convertresult {}.png'.format(iteration)
    array = rgb_array.astype(np.uint8)
    array = np.reshape(array, (h, w, 3))
    data = Image.fromarray(array)
    data.save(path_to_image)
    # data.show()
    return path_to_image

def lab2rgb(l, a, b):

    def lab_to_xyz(l, a, b):
        Y = (l + 16) / 116
        X = (a / 500) + Y
        Z = Y - (b / 200)

        def calc_xyz(var):
            if (var ** 3 > 0.008856):
                return (var ** 3)
            else:
                return (var - 16 / 116) / 7.787

        mod_Y, mod_X, mod_Z = calc_xyz(Y), calc_xyz(X), calc_xyz(Z)

        X = mod_X * 95.047
        Y = mod_Y * 100.000
        Z = mod_Z * 108.883

        return X, Y, Z

    var_X, var_Y, var_Z = lab_to_xyz(l, a, b)

    var_X = var_X / 100
    var_Y = var_Y / 100
    var_Z = var_Z / 100

    var_R = var_X * 3.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415
    var_B = var_X * 0.0557 + var_Y * -0.2040 + var_Z * 1.0570

    def calc_rgb(var):
        if (var > 0.0031308):
            return 1.055 * (var ** (1 / 2.4)) - 0.055
        else:
            return 12.92 * var

    R, G, B, = 255 * calc_rgb(var_R), 255 * \
        calc_rgb(var_G), 255 * calc_rgb(var_B)

    return  round(R, 8), round(G, 8), round(B, 8)


def rgb2lab(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    
    def make_xyz(R, G, B):
        X = R * 0.4124 + G * 0.3576 + B * 0.1805
        Y = R * 0.2126 + G * 0.7152 + B * 0.0722
        Z = R * 0.0193 + G * 0.1192 + B * 0.9505
        X = round(X, 8)
        Y = round(Y, 8)
        Z = round(Z, 8)

        
        X = float(X) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
        Y = float(Y) / 100.0          # ref_Y = 100.000
        Z = float(Z) / 108.883        # ref_Z = 108.883

        # X = float(X) / 96.422         # ref_X =  95.047   Observer= 2°, Illuminant= D50
        # Y = float(Y) / 100.0          # ref_Y = 100.000
        # Z = float(Z) / 82.521        # ref_Z = 108.883

        X, Y, Z = change_xyz(X), change_xyz(Y), change_xyz(Z)

        return X, Y, Z

    def change_xyz(value):
        if value > 0.008856:
            value = value ** (1/3.)
        else:
            value = (7.787 * value) + (16 / 116)

        return value

    def change_rgb(value):
        value = float(value) / 255

        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92

        return value * 100

    R, G, B = change_rgb(r), change_rgb(g), change_rgb(b)
    X, Y, Z = make_xyz(R, G, B)

    L = round((116 * Y) - 16, 8)
    a = round(500 * (X - Y), 8)
    b = round(200 * (Y - Z), 8)

    return (L, a, b)

def average_error(colorchecker):

    def calculate_error(real, current):
        if((real > 0 and current < 0 ) or (real < 0 and current > 0)):
            error = abs(real) + abs(current)
        else:
            error = abs(real - current)
        return error

    rgb_real = [[115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], 
    [103, 189, 170], [214, 126, 44],[80, 91, 166],[193, 90, 99],[94, 60, 108], [157, 188, 64],
    [224, 163, 46], [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31],[187, 86, 149],
    [8, 133, 161], [243, 243, 242], [200, 200, 200], [160, 160, 160],[122, 122, 121],[85, 85, 85],[52, 52, 52]]

    lab_real = np.apply_along_axis(rgb2lab, 1, rgb_real)
    # print('lab real: {} '.format(lab_real))


    lab_current = []
    for row in colorchecker.values:
        for values in row:
            rgb = [values[2], values[1], values[0]]
            new_lab = rgb2lab(rgb)
            lab_current.append(new_lab)

    # print('lab current: /n {} '.format(lab_current))

    error = 0
    color = 1

    stringetje = ""

    for ((real_l, real_a, real_b), (current_l, current_a, current_b)) in zip(lab_real, lab_current):
        
        error_l = calculate_error(real_l, current_l)
        error_a = calculate_error(real_a, current_a)
        error_b = calculate_error(real_b, current_b)
        error_channels = error_l + error_a + error_b
        error += error_channels

        stringetje += '{},'.format(round(error_channels))

        print('error {}: {}'.format(color, error_channels))
        color += 1

    stringetje = stringetje[:-1]
    stringetje += '\n'

    return error, stringetje


def optimize_transformation():
    f = open("Lab_error_per_color.txt", "w")
    f.write('1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24\n')
    f.close()

    with open("color_correction_brute_force_search.txt", "w") as output_file:
        output_file.write("iteration, error per color, red multiplier, green multiplier, blue multiplier\n")

    def evaluate_rgb_multiplier_correctness(multipliers, iteration_number):
        identifier = iteration_number #random.randint(0,100000)
        #print(identifier)
        r_multiplier, g_multiplier, b_multiplier = multipliers
      
        scan_tag = "color_exploration"
        scan_name = "{}_{}".format(scan_tag,identifier)
        photo_name = "{}-0_color_balanced_from_hdr.png".format(scan_name) # 
        global robot
        #robot.scan(project_name=scan_name, red_led_current=r_multiplier, green_led_current=g_multiplier, blue_led_current=b_multiplier, auto_focus=False, export_to_npy=False, process_point_cloud=False)
        robot.scan(project_name=scan_name, red_led_current=r_multiplier, green_led_current=g_multiplier, blue_led_current=b_multiplier, photos_to_project="r.png,g.png,b.png,ambient.png", export_to_npy=False, process_point_cloud=False, auto_focus=False, auto_exposure=False, hdr_exposure_times=[150.0])
        adjust_image_data, adjusted_image_colorchecker_average_values = find_macbeth(photo_name)
        cv.imwrite('transformed_colors_{}.png'.format(identifier), adjust_image_data)
        average_error_rgb, error_per_color = average_error(adjusted_image_colorchecker_average_values)

        f = open("Lab_error_per_color.txt", "a")
        f.write(error_per_color)
        f.close()

        return average_error_rgb

    def objective_function_evaluation(hypothesis):
        global iteration_number
        iteration_number += 1

        if iteration_number < 0:
            random_1 = random.uniform(0.0, 0.0001)
            random_2 = random.uniform(0.0, 0.0001)
            random_3 = random.uniform(0.0, 0.0001)
        else:
            random_1 = 0.0
            random_2 = 0.0
            random_3 = 0.0            

        r_guess = hypothesis[0]
        g_guess = hypothesis[1]
        b_guess = hypothesis[2]

        multipliers = [min((r_guess + random_1), 1.0), min((g_guess + random_2), 1.0), min((b_guess + random_3), 1.0)]

        print("\n\n\nHypothesis: {} (with perturbation ({},{},{}))".format(hypothesis,random_1,random_2,random_3), flush=True)

        # compute average error from Color Checker
        try:
            average_pixel_error_in_lab_space = evaluate_rgb_multiplier_correctness(multipliers, iteration_number)
        except KeyboardInterrupt:
            sys.exit(0)
        except Exception as err:
            print(Exception)
            average_pixel_error_in_lab_space = 10000.0
        #print("Error / color / channel lab space: {}".format(average_pixel_error_in_lab_space/72))
        print("Error in L*a*b color space: {}".format(average_pixel_error_in_lab_space))

        # penalize deviation from 1.0, because we don't want system to go over 1.0, and we also want to find the maximum possible power combination of R,G,B (which helps for scanning)  
        
        maximum_power = 1.00
        importance_of_above_power_restriction = 250.0
        red_deviation_above_max_power = importance_of_above_power_restriction * max(r_guess - maximum_power, 0.0)
        green_deviation_above_max_power = importance_of_above_power_restriction * max(g_guess - maximum_power, 0.0)
        blue_deviation_above_max_power = importance_of_above_power_restriction * max(b_guess - maximum_power, 0.0)
        sum_of_above_max_power_penalty = red_deviation_above_max_power + green_deviation_above_max_power + blue_deviation_above_max_power

        minimum_power = 0.85
        importance_of_below_power_restriction = 250.0
        red_deviation_below_max_power = importance_of_below_power_restriction * max(minimum_power - r_guess, 0.0)
        green_deviation_below_max_power = importance_of_below_power_restriction * max(minimum_power - g_guess, 0.0)
        blue_deviation_below_max_power = importance_of_below_power_restriction * max(minimum_power - b_guess, 0.0)
        sum_of_below_max_power_penalty = red_deviation_below_max_power + green_deviation_below_max_power + blue_deviation_below_max_power

        sum_of_power_deviation_penalty = sum_of_above_max_power_penalty + sum_of_below_max_power_penalty
 
        # typically we need to add a "regularizing coefficient" to properly weight different loss evaluation metrics that are combined
        # this is partially just considering how the different sizes of numbers are likely to be relative to each other
        print("Error from deviation in power: {}".format(sum_of_power_deviation_penalty))

        final_error_metric = average_pixel_error_in_lab_space / 24.0 #+ sum_of_power_deviation_penalty

        print("({}) Total error per color channel: {}\n\n\n".format(iteration_number, final_error_metric))

        with open("color_correction_brute_force_search.txt", "a") as output_file:
            output_file.write("{},{},{},{},{}\n".format(iteration_number,final_error_metric,r_guess,g_guess,b_guess))

        return final_error_metric


    #res = minimize(fun=objective_function_evaluation, x0=[0.75, 0.75, 0.75], args =[], options={'maxiter': 1000}, method='BFGS')
    red_range = [1.0] #slice(1.00,0.99, -0.01)
    green_range = [1.0] #slice(0.90,0.89,-0.01)
    blue_range = [1.0] #slice(0.81,0.80,-0.01)
    all_ranges = (red_range, green_range, blue_range)
    res = brute(func=objective_function_evaluation, ranges=((1.0,1.0),(0.8941,0.8941),(0.8031,0.8031)), args=(), finish=None)
    print(res)
    #final_hypotheses = res.x
    #final_multipliers = final_hypotheses[0], final_hypotheses[1], final_hypotheses[2]

    #return final_multipliers
    return res



if __name__ == '__main__':
    r_multiplier, g_multiplier, b_multiplier = optimize_transformation()




# Love.
# 3. French red wine and the best cheese of your dreams realized upon a wish.
# 4. The solver may have some legs to it, so let it get to it.
# 5. The reflectance to diffuse tradeoff problem is interesting! And 2 birds for one stone. The deep network approach is compelling.
# 6. In the end, cannot have perfect knowledge of color correction at a given angle without making assumption of reflection.
# 7. It would help to have an hour in the morning to prioritize next steps.











# def write_results(colorchecker, image, filename=None):
#     plt.use('Agg')
#     mes = 'color_index,pixel_x,pixel_y,l,a,b,real_l,real_a,real_b\n'

#     real_lab_colors = []
#     for dict_item in colorchecker_truecolors():
#         for key in dict_item:
#             if (key == 'lab'):
#                 real_lab_colors.append(dict_item[key])
#     real_rgb_colors = []
#     for dict_item in colorchecker_truecolors():
#         for key in dict_item:
#             if (key == 'rgb'):
#                 real_rgb_colors.append(dict_item[key])

#     n = 0
#     for corners in calculate_edges(colorchecker):
#         (real_l, real_a, real_b) = real_lab_colors[n]
#         (x_min, x_max, y_min, y_max) = corners
#         for y in range(int(y_min), int(y_max)):
#             for x in range(int(x_min), int(x_max)):

#                 l_point, a_point, b_point = lab_image[x, y]

#                 error_l = abs(l_point - real_l)
#                 error_a = abs(a_point - real_a)
#                 error_b = abs(b_point - real_b)

#                 if(a_point > 0 and real_a < 0):
#                     error_a = a_point + abs(real_a)
#                 elif(a_point < 0 and real_a > 0):
#                     error_a = abs(a_point) + real_a
#                 else:
#                     error_a = abs(a_point) - abs(real_a)

#                 if(b_point > 0 and real_b < 0):
#                     error_b = b_point + abs(real_b)
#                 elif(b_point < 0 and real_b > 0):
#                     error_b = abs(b_point) + real_b
#                 else:
#                     error_b = abs(b_point) - abs(real_b)

#                 mes += '{},{},{},{},{},{},{},{},{}\n'.format(
#                     (n + 1), x, y, l_point, a_point, b_point, real_l, real_a, real_b)
#         n += 1

#     if filename is None:
#         print(mes)
#     else:
#         with open(filename, 'w+') as f:
#             f.write(mes)
