# Copyright 2020 by Utku Karadeniz, Language Technology Lab.
# All rights reserved.
# This file is part of Ruling Line Removal Project at the Language Technology Lab, University of Duisburg-Essen.

import sys
import numpy as np
import cv2
import Detect_RulingLines as drl
from Tools import check_ForGrayImage, make_GrayImage
import config as cnf


class RemovalType:
    Brutal = 0
    MajorityVoting = 1
    LineIterator = 2


class RulingLineRemoval:
    def __init__(self, img, filename):
        # checking if the image is gray scaled
        if not check_ForGrayImage(img):
            print("Please input a gray-scaled image to RulingLineRemoval Constructor")
            return -1

        # make the cropped image binary using a threshold
        ret, self.original_page = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # set the height and the width of the original page which will be used in other functions
        self.width = np.size(self.original_page, 1)
        self.height = np.size(self.original_page, 0)
        self.filename = filename


    def calculate_line_distance(self, plot_houghlines=0, print_lines=0, plot_strongest_2_lines=0):
        """Function to calculate the line distance of a given image
            display all the found lines and the strongest two, as well as print all the lines
            Parameters:
                plot_houghlines : binary value to determine whether the user wants to view all the found ruling lines
                for each window in separate images
                print_lines     : binary value to determine whether the user wants to print all the found ruling lines
                by the Hough Transform function for each window
                plot_strongest_2_lines: binary value to determine whether the user wants view the 2 strongest
                ruling lines for each window in separate images
            Returns:
                line_distance  : a floating point number for the estimated line distance
            For the given image, the line distance for 4 windows are calculated and the median of these calculations
            are estimated to be the line distance
        """
        # STEP 1: Make the Windows S32, S35, S52 and S55
        # calculate 1/5 of the width
        width_5 = int(self.width / 5)

        # calculate the lost part from the top of the page
        lost = int(self.height - int(self.height / width_5) * width_5)

        # calculate the boundaries of the windows
        height_start_W3x = lost + 2 * width_5
        height_end_W3x = lost + 3 * width_5
        height_start_W5x = lost + 4 * width_5
        height_end_W5x = lost + 5 * width_5

        width_start_Wx2 = width_5
        width_end_Wx2 = 2 * width_5
        width_start_Wx4 = 3 * width_5
        width_end_Wx4 = 4 * width_5

        # make the windows S32, S35, S52 and S55
        W32 = self.original_page[height_start_W3x:height_end_W3x, width_start_Wx2:width_end_Wx2]
        W34 = self.original_page[height_start_W3x:height_end_W3x, width_start_Wx4:width_end_Wx4]
        W52 = self.original_page[height_start_W5x:height_end_W5x, width_start_Wx2:width_end_Wx2]
        W54 = self.original_page[height_start_W5x:height_end_W5x, width_start_Wx4:width_end_Wx4]

        # STEP 2: Use the Hough transform to find the strongest two lines in the windows
        W32_1, W32_2 = drl.find_HoughLines("W32", W32, print_lines, plot_houghlines, plot_strongest_2_lines)
        W34_1, W34_2 = drl.find_HoughLines("W34", W34, print_lines, plot_houghlines, plot_strongest_2_lines)
        W52_1, W52_2 = drl.find_HoughLines("W52", W52, print_lines, plot_houghlines, plot_strongest_2_lines)
        W54_1, W54_2 = drl.find_HoughLines("W54", W54, print_lines, plot_houghlines, plot_strongest_2_lines)

        # STEP 3: Calculate the line distance
        # calculate the distance between two strongest lines found in each window
        diff1 = (abs(W32_1[0] - W32_2[0]))
        diff2 = (abs(W34_1[0] - W34_2[0]))
        diff3 = (abs(W52_1[0] - W52_2[0]))
        diff4 = (abs(W54_1[0] - W54_2[0]))

        # make an array containing the differences
        diff = np.array([diff1, diff2, diff3, diff4])

        # determine a temp value which serves as an estimate for the line distance
        temp = self.height / 17

        # find the values in diff which is less than half of the temp and remove them from the array
        # this takes cares of the cases where the found lines are on the same row
        result = np.where(diff < temp / 2)
        diff = np.delete(diff, result)

        # if the diff represents a difference of more than two line distance, divide by three
        diff[diff > temp * 2.05] = diff[diff > temp * 2.05] / 3

        # if the diff represents a difference of more than one line distance, divide by half
        diff[diff > temp * 1.05] = diff[diff > temp * 1.05] / 2

        # determine the median of the array as line distance
        line_distance = np.median(diff)

        return line_distance


    def find_ruling_lines(self, line_distance, draw_ruling_lines=0):
        """
            Function to find the real ruling lines of the image
            Parameters:
                line_distance   : the calculated line distance (return value of the function "calculate_line_distance")
                draw_ruling_lines: binary value to determine whether the user wants the found ruling lines
                to be written on image with the given filename
            Returns:
                ruling_lines_start: a list containing all the y-coordinates of the found ruling lines from the left
                ruling_lines_end  : a list containing all the y-coordinates of the found ruling lines from the right
            For the given image, 4 windows are...(explanation of the algorithm)
        """

        # make a copy of the original page
        original_copy = self.original_page.copy()

        # invert the image
        original_copy = 255 - original_copy

        # make the image binary
        img_binary = original_copy / 255

        # percentage_width to crop image from the left and right
        percentage_width = 0.25

        # crop image from the left and right end
        img_binary_start = img_binary[:, 0:round(self.width * percentage_width)]
        img_binary_end = img_binary[:, (round(self.width * (1 - percentage_width))-0):(self.width - 200)]

        # find the start points of the ruling lines from the left
        ruling_lines_start = drl.find_StartEndPoints(img_binary_start, line_distance)

        if draw_ruling_lines:
            self.draw_ruling_lines(ruling_lines_start, "start", original_copy)

        # find the start points of the ruling lines from the right
        ruling_lines_end = drl.find_StartEndPoints(img_binary_end, line_distance)

        if draw_ruling_lines:
            self.draw_ruling_lines(ruling_lines_end, "end", original_copy)

        return ruling_lines_start, ruling_lines_end


    def draw_ruling_lines(self, ruling_lines, startorend, original_page):
        """ Functions draw the ruling lines
            Parameters:
                ruling_lines    : y-coordinate of the found ruling lines
                startorend      : left or right part of the image
                original_page   : original cropped image
            Returns:
                -
        """

        # creating new images to draw the lines in
        # cdst - ruling lines on top of the original image
        # cdst2 - ruling lines on white background
        # cdst3 - ruling lines on top of the binary image
        cdst = cv2.cvtColor(original_page, cv2.COLOR_GRAY2BGR)
        cdst2 = 255 - cdst

        line_thickness = 1

        for i in range(len(ruling_lines)):
            current = ruling_lines[i]
            pt1 = (0, current)
            pt2 = (np.size(original_page, 1), current)
            cv2.line(cdst, pt1, pt2, (0, 255, 0), line_thickness)  # RGB
            cv2.line(cdst2, pt1, pt2, (0, 255, 0), line_thickness)  # RGB

        cv2.imwrite(cnf.folder_vis + self.filename + "_" + startorend + '_with_text.png', cdst)
        cv2.imwrite(cnf.folder_vis + self.filename + "_" + startorend + '_on_binary_image.png', cdst2)


    def create_lineIterator(self, P1, P2, img):

        """Produces an array that consists of the coordinates and intensities of each pixel in a line between two points

        Parameters:
            -P1: a numpy array that consists of the coordinate of the first point (x,y)
            -P2: a numpy array that consists of the coordinate of the second point (x,y)
            -img: the image being processed

        Returns:
            -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
        """
        # define local variables for readability
        imageH = img.shape[0]
        imageW = img.shape[1]
        P1X = P1[0]
        P1Y = P1[1]
        P2X = P2[0]
        P2Y = P2[1]

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        dX = P2X - P1X
        dY = P2Y - P1Y
        dXa = np.abs(dX)
        dYa = np.abs(dY)

        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        negY = P1Y > P2Y
        negX = P1X > P2X
        if P1X == P2X:  # vertical line segment
            itbuffer[:, 0] = P1X
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
        elif P1Y == P2Y:  # horizontal line segment
            itbuffer[:, 1] = P1Y
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
        else:  # diagonal line segment
            steepSlope = dYa > dXa
            if steepSlope:
                slope = dX.astype(np.float32) / dY.astype(np.float32)
                if negY:
                    itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
            else:
                slope = dY.astype(np.float32) / dX.astype(np.float32)
                if negX:
                    itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

        # Remove points outside of image
        colX = itbuffer[:, 0]
        colY = itbuffer[:, 1]
        itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

        # Get intensities from img ndarray
        itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

        return itbuffer


    def line_removal_brutal(self, ruling_lines_start, ruling_lines_end, window_start, window_end):
        """ Function to remove the ruling lines from the image using the brutal algorithm where all the pixels
            within the given window and line thickness is converted to the background color
            Parameters:
                ruling_lines_start: a list containing all the y-coordinates of the found ruling lines from the left
                ruling_lines_end  : a list containing all the y-coordinates of the found ruling lines from the right
                window_start    : integer determining the higher boundary of the window the removal algorithm affects
                window_end      : integer determining the lower boundary of the window the removal algorithm affects
            Returns:
                removed_pixels: a binary inverted image having white pixels as all the removed pixels and
                black pixels as the rest
        """
        # copy the original page
        original_page_for_removal = np.copy(self.original_page)

        # offset, so that lines with text that extend beyond ruling lines are unaffected by the removal
        offset = -220

        # Brutal Line Removal
        for i in range(len(ruling_lines_start)):
            lines = original_page_for_removal[ruling_lines_end[i] - window_start:ruling_lines_start[i] + window_end, :offset]
            lines[lines != 255] = 255

        filepath_rl_removed_page = cnf.folder_res + self.filename + "_brutal" + ".png"
        # image without the ruling lines is written to a file
        cv2.imwrite(filepath_rl_removed_page, original_page_for_removal)

        # a binary image containing the pixels removed by the chosen algorithm is calculated
        removed_pixels = original_page_for_removal - self.original_page

        # uncomment below to write a binary image containing all the removed_pixels onto the vis directory
        # filepath_rl_removed_px = cnf.folder_vis + self.filename + "_brutal_removed_pixels" + ".png"
        # cv2.imwrite(filepath_rl_removed_px, removed_pixels)

        return removed_pixels


    def line_removal_majorityVoting(self, ruling_lines_start, ruling_lines_end, line_thickness, window_start, window_end):
        """ Function to remove the ruling lines from the image using the majority voting algorithm where all the pixels
            within the given window are converted to the background color if the number of black pixels in the window
            is less than the line thickness
            Parameters:
                ruling_lines_start: a list containing all the y-coordinates of the found ruling lines from the left
                ruling_lines_end  : a list containing all the y-coordinates of the found ruling lines from the right
                line_thickness  : integer determining the line thickness the removal algorithm assumes
                window_start    : integer determining the higher boundary of the window the removal algorithm affects
                window_end      : integer determining the lower boundary of the window the removal algorithm affects
            Returns:
                removed_pixels: a binary inverted image having white pixels as all the removed pixels and
                black pixels as the rest
        """
        # copy the original page
        original_page_for_removal = np.copy(self.original_page)

        # offset, so that lines with text that extend beyond ruling lines are unaffected by the removal
        offset = -220

        # Majority voting
        for i in range(len(ruling_lines_start)):
            lines = original_page_for_removal[ruling_lines_end[i] - window_start:ruling_lines_start[i] + window_end, :-220]
            for j in range(np.size(lines, 1)):
                vertical = lines[:, j]
                if np.count_nonzero(vertical == 0) <= line_thickness:
                    vertical[vertical == 0] = 255

        filepath_rl_removed_page = cnf.folder_res + self.filename + "_majority_voting" + ".png"
        # image without the ruling lines is written to a file
        cv2.imwrite(filepath_rl_removed_page, original_page_for_removal)

        # a binary image containing the pixels removed by the chosen algorithm is calculated
        removed_pixels = original_page_for_removal - self.original_page

        # uncomment below to write a binary image containing all the removed_pixels onto the vis directory
        # filepath_rl_removed_px = cnf.folder_vis + self.filename + "_majorityVoting_removed_pixels" + ".png"
        # cv2.imwrite(filepath_rl_removed_px, removed_pixels)

        return removed_pixels


    def line_removal_lineIterator(self, ruling_lines_start, ruling_lines_end, line_thickness, window_start, window_end):
        """ Function to remove the ruling lines from the image using the line iterator algorithm where a lines is
            drawn all the pixels
            within the given window are converted to the background color if the number of black pixels in the window
            is less than the line thickness
            Parameters:
                ruling_lines_start: a list containing all the y-coordinates of the found ruling lines from the left
                ruling_lines_end  : a list containing all the y-coordinates of the found ruling lines from the right
                line_thickness  : integer determining the line thickness the removal algorithm assumes
                window_start    : integer determining the higher boundary of the window the removal algorithm affects
                window_end      : integer determining the lower boundary of the window the removal algorithm affects
            Returns:
                removed_pixels: a binary inverted image having white pixels as all the removed pixels and
                black pixels as the rest
        """
        # copy the original page
        original_page_for_removal = np.copy(self.original_page)

        # Line Iterator
        for i in range(len(ruling_lines_start)):
            start_point = np.array([0, ruling_lines_start[i]])
            # offset, so that lines with text that extend beyond ruling lines are unaffected by the removal
            offset = 220
            end_point = np.array([self.width - offset, ruling_lines_end[i]])
            end_point = np.array([self.width, ruling_lines_end[i]])
            points = self.create_lineIterator(start_point, end_point, original_page_for_removal)
            for j in range(len(points)):
                x = int(points[j][0])
                y = int(points[j][1])
                vertical = original_page_for_removal[y - window_start: y + window_end, x]
                if np.count_nonzero(vertical == 0) <= line_thickness:
                    vertical[vertical == 0] = 255

        filepath_rl_removed_page = cnf.folder_res + self.filename + "_line_iterator" + ".png"
        # image without the ruling lines is written to a file
        cv2.imwrite(filepath_rl_removed_page, original_page_for_removal)

        # a binary image containing the pixels removed by the chosen algorithm is calculated
        removed_pixels = original_page_for_removal - self.original_page

        # uncomment below to write a binary image containing all the removed_pixels onto the vis directory
        # filepath_rl_removed_px = cnf.folder_vis + self.filename + "_lineIterator_removed_pixels" + ".png"
        # cv2.imwrite(filepath_rl_removed_px, removed_pixels)

        return removed_pixels


    def visualization(self, removed_pixels, removal_algorithm):
        """ Function to visıalize all the removed pixels
            Parameters:
                removed_pixels: a binary inverted image having white pixels as all the removed pixels and
                black pixels as the rest
                removal_algorithm  : the name of the removal_algorithm being used
            Returns:
                -
            The visualization of the removed pixels are shown in a newly constructed image where handwriting is blue,
            removed pixels are blue and background is black.
        """
        # turn gray image to a colorful image
        original_page_with_color = cv2.cvtColor(self.original_page, cv2.COLOR_GRAY2BGR)

        # invert the image
        original_page_with_color = 255 - original_page_with_color

        # split the image
        b, g, r = cv2.split(original_page_with_color)

        # create a black background for the visualization
        zeros = np.copy(self.original_page)
        zeros[zeros != 0] = 0

        # merge the image to create the visualization
        removed_pixels_visualization = cv2.merge((b, zeros, removed_pixels))

        removal_algorithm = removal_algorithm.replace(" ", "_")
        # write the file into a png file
        filepath_visualization = cnf.folder_vis + self.filename + "_" + removal_algorithm + "_vis" + ".png"
        cv2.imwrite(filepath_visualization, removed_pixels_visualization)


    def evaluation(self, removed_pixels, removal_algorithm):
        """ Function to calculate the statistics for the chosen removal algorithm
            Parameters:
                removed_pixels: a binary inverted image having white pixels as all the removed pixels and
                black pixels as the rest
                removal_algorithm  : the name of the removal_algorithm being used
            Returns:
                -
            The evaluation of the used algorithm is shown by printing out the statistics as well as constructing
            a new visualization image where green denotes correctly removed pixels, red denotes incorrectly
            removed pixels and blue denotes pixels that should have been removed but were not.
        """
        # read the image with removed ruling lines
        img_without_ruling_lines = cv2.imread(cnf.folder_in + 'original_page_no_line.png', 0)

        if not check_ForGrayImage(img_without_ruling_lines):
            print("The original_page_no_line.png is not gray image")
            print("Transforming the image to gray scaled...")
            make_GrayImage(img_without_ruling_lines)
            print("Transformation complete")

        # binarize the image
        ret, img_without_ruling_lines = cv2.threshold(img_without_ruling_lines, 127, 255, cv2.THRESH_BINARY)

        # invert the read image
        img_without_ruling_lines = 255 - img_without_ruling_lines

        # determine a percentage_height to get the first 7 lines of a page
        percentage_height = 0.345

        # crop the first 7 lines of the page with removed ruling lines
        img_without_ruling_lines = img_without_ruling_lines[0:round(self.height * percentage_height), :]
        filepath_img_without_ruling_lines = cnf.folder_vis + self.filename + "_" + "Original-RulerLine" + ".png"

        # make a copy of the original image
        original_page_copy = cv2.imread(cnf.folder_out + "split\\" + self.filename + ".png", 0)

        # invert the image - values of 0 or 255
        original_page_copy = 255 - original_page_copy

        # first 7 lines of the copied original image
        img_binary_original = original_page_copy[0:round(self.height * percentage_height), :]
        filepath_img_binary_original = cnf.folder_vis + self.filename + "_" + "Original" + ".png"

        if img_without_ruling_lines.shape != img_binary_original.shape:
            print(img_binary_original.shape)
            print(img_without_ruling_lines.shape)
            sys.exit("The cropped-out original page and img_without_ruling_lines are not the same size!")

        # subtract original image with the 99% image to get the ruling lines only
        ruling_lines = img_binary_original - img_without_ruling_lines

        # get the first 7 lines of the removed pixels
        removed_pixels_first_7 = removed_pixels[0:round(self.height * percentage_height), :]

        ret, removed_pixels_first_7 = cv2.threshold(removed_pixels_first_7, 127, 255, cv2.THRESH_BINARY)

        # bitwise_and the ruling lines with the removed pixel image to get the correctly identified pixels
        correct_pixels = cv2.bitwise_and(removed_pixels_first_7, ruling_lines)

        # subtract ruling lines from the correctly identified pixels to get the missed lines
        missed_lines = ruling_lines - correct_pixels

        # subtract removed pixel image from the correctly identified pixels to get the rest - wrongly identified pixels
        different_pixels = removed_pixels_first_7 - correct_pixels

        # counting the pixels for each image
        no_pixels_removed = np.count_nonzero(removed_pixels_first_7)
        no_pixels_ruling_lines = np.count_nonzero(ruling_lines)
        correctly_removed = np.count_nonzero(correct_pixels)
        missed_lines_count = np.count_nonzero(missed_lines == 255)
        wrongly_removed = np.count_nonzero(different_pixels)

        # displaying the statistics
        print("\nNumber of Ruling Line pixels         = " + str(no_pixels_ruling_lines))
        print("Number of pixels removed             = " + str(no_pixels_removed))
        print("Correctly Removed Ruling Line Pixels = " + str(correctly_removed/no_pixels_ruling_lines*100))
        # print("Missed Ruling Line Pixels            = " + str(missed_lines_count / no_pixels_ruling_lines * 100))
        print("Correctly Removed Pixels             = " + str(correctly_removed / no_pixels_removed * 100))
        # print("Incorrectly Removed Pixels           = " + str(wrongly_removed / no_pixels_removed*100))


        # making a visualisation where green: correct_pixels - red: different_pixels - blue: missed_lines
        evaluation_vis = cv2.merge((missed_lines, correct_pixels, different_pixels))
        filepath_evaluation_vis = cnf.folder_vis + self.filename + "_" + removal_algorithm + "_eval" + ".png"
        cv2.imwrite(filepath_evaluation_vis, evaluation_vis)


    @staticmethod
    def processing(img, filename, removal_algorithm, line_thickness, window_start, window_end, visualization="no",
                   evaluation="no", plot_houghlines=0, print_lines=0, plot_strongest_2_lines=0, draw_ruling_lines=0):
        """ Processing function that calls all the necessary functions of the class
            Parameters:
                img: gray scaled image
                filename: filename of the page being worked on
                savepath: directory where the results should be saved
                removal_algorithm: string containing the filename of the desired removal algorithm to be used
                line_thickness  : integer determining the line thickness the removal algorithm assumes
                window_start    : integer determining the higher boundary of the window the removal algorithm affects
                window_end      : integer determining the lower boundary of the window the removal algorithm affects
                visualization   : string to decide whether the visualization of removed pixels are to be displayed or not
                evaluation      : string to decide whether the evaluation statistics are to be displayed or not
                plot_houghlines : binary value to determine whether the user wants to view all the found ruling lines
                for each window in separate images
                print_lines     : binary value to determine whether the user wants to print all the found ruling lines
                by the Hough Transform function for each window
                plot_strongest_2_lines: binary value to determine whether the user wants view the 2 strongest
                ruling lines for each window in separate images
            Returns:
                -
        """

        # check if an existing removal algorithm was chosen
        if removal_algorithm > RemovalType.LineIterator or removal_algorithm < 0:
            print("Please input one of the existing removal algorithms:  0(brutal), 1(majorityVoting) or 2(lineIterator)")
            return -1

        # calling the constructor of the class to initialize everything
        RLR = RulingLineRemoval(img, filename)

        # calculating the line distance
        line_distance = RLR.calculate_line_distance(plot_houghlines, print_lines, plot_strongest_2_lines)
        print("The estimated line distance is: " + str(line_distance))

        # finding the ruling lines
        ruling_lines_start, ruling_lines_end = RLR.find_ruling_lines(line_distance, draw_ruling_lines)
        if len(ruling_lines_start) != len(ruling_lines_end):
            print("ups, seems like the number of ruling lines found at the left and the right end are not the same,"
                  "you might wanna check in code lines: 127 - 131")
            return -1

        # removing the ruling lines with the chosen algorithm
        if removal_algorithm == RemovalType.Brutal:
            removal_algorithm = "Brutal"
            removed_pixels = RLR.line_removal_brutal(ruling_lines_start, ruling_lines_end, window_start, window_end)

        elif removal_algorithm == RemovalType.MajorityVoting:
            removal_algorithm = "MajorityVoting"
            removed_pixels = RLR.line_removal_majorityVoting(ruling_lines_start, ruling_lines_end,
                                                             line_thickness, window_start, window_end)
        elif removal_algorithm == RemovalType.LineIterator:
            removal_algorithm = "LineIterator"
            removed_pixels = RLR.line_removal_lineIterator(ruling_lines_start, ruling_lines_end,
                                                           line_thickness, window_start, window_end)

        # if requested visualising removed pixels
        if visualization.lower() == "yes":
            RLR.visualization(removed_pixels, removal_algorithm)

        # if requested the evaluation statistics of the algorithm is provided
        if evaluation.lower() == "yes":
            RLR.evaluation(removed_pixels, removal_algorithm)

