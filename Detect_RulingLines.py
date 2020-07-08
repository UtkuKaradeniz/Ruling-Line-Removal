# Copyright 2020 by Utku Karadeniz, Language Technology Lab.
# All rights reserved.
# This file is part of Ruling Line Removal Project at the Language Technology Lab, University of Duisburg-Essen.

import numpy as np
import cv2
from scipy.signal import argrelextrema
from Tools import check_ForGrayImage, make_GrayImage


import numpy as np
import cv2
from scipy.signal import argrelextrema
from Tools import check_ForGrayImage, make_GrayImage


def find_HoughLines(img_name, img, print_lines=0, plot_houghlines=0, plot_strongest_2_lines = 0):
    """Function to calculate and return the two of the strongest lines found for a given window, optionally one could
        display all the found lines and the strongest two, as well as print all the lines
        Parameters:
            img_name        : name of the window the images will be outputted
            img             : a gray-scale image
            print_lines     : binary value to determine whether the user wants to print all the found ruling lines
            by the Hough Transform function for each window
            plot_houghlines : binary value to determine whether the user wants to view all the found ruling lines
            for each window in separate images
            plot_strongest_2_lines: binary value to determine whether the user wants view the 2 strongest
            ruling lines for each window in separate images
        Returns:
            first_strongest  : array containing the distance to upper left corner of the window and
                                angle of the strongest line
            second_strongest : array containing the distance to upper left corner of the window and
                                angle of the second strongest line
    """
    # use Edge Detection to get an image for Hough Tranfrom
    edges = cv2.Canny(img, 50, 150, apertureSize=3)

    # optionally one could uncomment the below line to use sobely instead of Canny-Edge Detection
    # edges = cv2.Sobel(img, cv2.CV_8UC1, 0, 1, ksize=5)

    # threshold for detecting ruling lines is one 3rd of the width of the windows
    threshold = (np.size(img, 1)//3)

    # Hough Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=threshold)

    # calling the plot function and printing the lines
    if print_lines:
        print(lines)
    if plot_houghlines or plot_strongest_2_lines:
        plot_HoughLines(img_name, img, lines, plot_houghlines, plot_strongest_2_lines)

    return np.squeeze(lines[0]), np.squeeze(lines[1])


def find_StartEndPoints(img_binary, line_distance):
    """ Functions to find the ruling lines and upon request make images containing the found ruling lines
    Parameters:
        img_binary      : binary input image
        line_distance   : estimated line distance
    Returns:
        ruling_lines    : a list containing all the y-coordinate of the found ruling lines
    """

    # Horizontal Projection
    horizontal_sum = (cv2.reduce(img_binary, 1, cv2.REDUCE_SUM, dtype=cv2.CV_64FC1))

    kernel_size = 51
    # Blurring to smooth the horizontal projection
    signal_smooth = cv2.GaussianBlur(horizontal_sum, (kernel_size, kernel_size), 0, 0)

    # Find the local extremum of the signal
    max_indexes = argrelextrema(signal_smooth, np.greater)
    maxim = max_indexes[0]

    # interval for searching ruling lines between peaks
    search_peak_interval = 40
    lower_bound = int(line_distance) - search_peak_interval
    upper_bound = int(line_distance) + search_peak_interval + 1
    search_peak = np.arange(start=lower_bound, stop=upper_bound)

    # interval for finding the real maximum of the signal - around 60 is the ideal number
    if maxim[0] < 60:
        search_max_interval = maxim[0]
    else:
        search_max_interval = 59

    # a list to store the y-axis position of the ruling lines
    ruling_lines = []

    # setting current to be the first maximum
    i = 0
    current = maxim[0]

    # loop to find the real ruling lines
    while i < len(maxim)-1 and current <= maxim[-1] and len(ruling_lines) < 22:
        mask = np.isin(maxim, current + search_peak)
        if np.any(mask):
            current = current-search_max_interval+np.argmax(horizontal_sum[current-search_max_interval:current+search_max_interval])
            ruling_lines.append(current)
            current = maxim[mask][0]
            k = current
            if k == maxim[-1]:
                current = current - search_max_interval + np.argmax(horizontal_sum[current - search_max_interval:current+search_max_interval])
                ruling_lines.append(current)
                break
        else:
            del ruling_lines[:]
            i += 1
            current = maxim[i]

    return ruling_lines


def plot_HoughLines(img_name, img, lines, plot_houghlines=0, plot_strongest_2_lines=0):

    if not check_ForGrayImage(img):
        make_GrayImage(img)

    # image to draw the lines on
    cdst = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cdst2 = np.copy(cdst)

    if plot_houghlines:
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                cv2.line(cdst, pt1, pt2, (0, 0, 255), 1)

        cv2.imshow("Source - " + img_name + "", img)
        cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
        cv2.waitKey(0)

    # if requested displaying the strongest two lines
    if plot_strongest_2_lines:
        for i in range(2):
            # create a tuple for the starting point of strongest line
            strongest_start = (int((np.cos(lines[i][0][1]) * lines[i][0][0]) + 1000 * (-np.sin(lines[i][0][1]))),
                               int(np.sin(lines[i][0][1]) * lines[i][0][0] + 1000 * (np.cos(lines[i][0][1]))))

            # create a tuple for the end point of strongest line
            strongest_end = (int((np.cos(lines[i][0][1]) * lines[i][0][0]) - 1000 * (-np.sin(lines[i][0][1]))),
                             int(np.sin(lines[i][0][1]) * lines[i][0][0] - 1000 * (np.cos(lines[i][0][1]))))

            # drawing the found ruling lines onto the image
            cv2.line(cdst2, strongest_start, strongest_end, (0, 255, 0), 1)

        cv2.imshow("Source - " + img_name + "", cdst2)
        cv2.imshow("2 Strongest Lines (in green) - Standard Hough Line Transform", cdst2)
        cv2.waitKey(0)

