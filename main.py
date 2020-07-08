# Copyright 2020 by Utku Karadeniz, Language Technology Lab.
# All rights reserved.
# This file is part of Ruling Line Removal Project at the Language Technology Lab, University of Duisburg-Essen.

import sys
import numpy as np
import cv2
import config as cnf
from Tools import make_GrayImage
from Remove_RulingLines import RulingLineRemoval


def main():

    # choose file!
    filename = "sample-1.png"
    
    preprocessed_img = Preprocessing(filename=filename)
    right_page, left_page = preprocessed_img.auto_crop()
    print("Pre-processing finished!\n")

    # choose the removal algorithm!
    # 0 = brutal; 1 = majorityVoting; 2 = lineIterator
    removal_algorithm = 2

    # line_thickness is the assumed line_thickness in removal algorithm
    # window_start and window_end is the assumed window in removal algorithm

    # calling the processing function for right page
    print("Stats for right page:")
    filename_right_page = filename[:-4] + "_rp"
    RulingLineRemoval.processing(img=right_page, filename=filename_right_page, removal_algorithm=removal_algorithm,
                line_thickness=5, window_start=3, window_end=6, visualization="yes", evaluation="yes",
                plot_houghlines=0, print_lines=0, plot_strongest_2_lines=0, draw_ruling_lines=0)
    
    # calling the processing function for left page
    print("\nStats for left page:")
    filename_left_page = filename[:-4] + "_lp"
    RulingLineRemoval.processing(img=left_page, filename=filename_left_page, removal_algorithm=removal_algorithm,
                line_thickness=3, window_start=3, window_end=4, visualization="yes", evaluation="no",
                 plot_houghlines=0, print_lines=0, plot_strongest_2_lines=0, draw_ruling_lines=1)


class Preprocessing:
    
    def __init__(self, filename):
        # gray_scale_read = 0
        img = cv2.imread(str(cnf.folder_in + filename), cv2.IMREAD_GRAYSCALE)

        if img is None:
            sys.exit("Img is empty")

        # make image to gray image
        img = make_GrayImage(img)
        self.filename_stem = filename[:-4]
        self.img = img
        self.savepath = cnf.folder_out + "split/"

    def auto_crop(self):
        """ Automatically crops a given image into right and left image
            Parameters:
                -
            Returns:
                right_page  : right page of the given image
                left_page   : left page of the given image
        """

        # percentage for determining where to search for the black squares
        percentage_for_searching_contours = 0.1

        width = np.size(self.img, 1)

        # calculating the boundaries
        width_bound = round(width * percentage_for_searching_contours)

        # crop the images
        left = self.img[:, 0:width_bound]
        right = self.img[:, width - width_bound:]

        # convert the images to colored
        left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        right = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)

        # find the squares
        result_left = self.find_the_squares(left)
        result_right = self.find_the_squares(right)


        # cropping for left image
        height_start = result_left[-1][1]
        height_end = result_left[0][1]

        # below parameters were heuristically chosen
        width_start = result_left[-1][0] + result_left[-1][2] + 135
        width_end = width_start + 1900

        left_page = self.img[height_start:height_end, width_start:width_end]

        filepath_left_page = self.savepath + self.filename_stem + "_lp" + ".png"
        cv2.imwrite(filepath_left_page, left_page)

        # cropping for right image (currently 1st page)
        if self.filename_stem[-1] == "0":
            height_end = result_right[0][1] - 1
            height_start = height_end - 2417
            a = width - width_bound + result_right[0][0]
            width_end = a + result_right[0][2] + 60
            width_start = a - 2085
            right_page = self.img[height_start:height_end, width_start:width_end]
        elif self.filename_stem[-1] == "1":
            height_start = result_left[-1][1]
            height_end = result_left[0][1]
            a = width - width_bound + result_right[0][0] + 1
            width_end = a + result_right[0][2] - 120
            width_start = a - 1826
            right_page = self.img[height_start:height_end, width_start:width_end]

            # removing any unwanted printing on the left-most side of the page
            # sum all the non-white pixels in y-axis for the first 30 columns
            count = (right_page[:, 0:30] < 255).sum(axis=0)

            # find the array containing the first full white column
            firs_whitepx = np.where(count == 0)[0]

            # if such a column exists and is not the first or the last, then
            # turn all pixels to background from the column till left corner
            if firs_whitepx.size != 0:
                if firs_whitepx[0] != 29 and firs_whitepx[0] != 0:
                    right_page[:, 0:firs_whitepx[0]] = 255

            # turning the barcode in the upper right corner to background color
            # right_page_width = np.size(right_page, 1)
            # right_page[0:297, -241:right_page_width] = 255

        filepath_right_page = self.savepath + self.filename_stem + "_rp" + ".png"
        cv2.imwrite(filepath_right_page, right_page)

        return right_page, left_page


    def find_the_squares(self, img):
        """ Finds the black squares at the sides of the images
            Parameters:
                img: gray scaled image
            Returns:
                result: an array of arrays containing the found squares where
                each of the element is represented as [x y w h] - x and y coordinate of the upper left point of the
                rectangle, the width and height of the square
        """
        to_draw = img.copy()
        # the image has to be colored
        result = []
        font = cv2.FONT_HERSHEY_COMPLEX
        edged = cv2.Canny(img, 50, 200)
        if cv2.__version__.startswith('3.'):
            (_, contours, _) = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        else:
            (contours, _) = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # _,contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #_, contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # loop for finding and drawing the contours
        for cnt in contours:
            # find all the rectangles
            x, y, w, h = cv2.boundingRect(cnt)
            # draw all the rectangles
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

            # an estimated value for the area of the black sqaures
            estimated_area = 65 * 64
            deviation = 250
            test = np.arange(start=estimated_area - deviation, stop=estimated_area + deviation + 1)
            real_area = w * h

            if np.any(np.isin(test, real_area)):
                # print(str(x) + "," + str(y) + "," + str(w) + "," + str(h))
                # to_draw[y][x] = (255, 0, 255)
                # cv2.imwrite(cnf.folder_out+"countours.png", to_draw)
                result.append([x, y, w, h])

        # removing the duplicate entries in result
        result = [result[i] for i in range(len(result)) if i == 0 or result[i] != result[i - 1]]
        return result


if __name__ == '__main__':
    main()

