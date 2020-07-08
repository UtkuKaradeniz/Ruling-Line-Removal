# Copyright 2020 by Utku Karadeniz, Language Technology Lab.
# All rights reserved.
# This file is part of Ruling Line Removal Project at the Language Technology Lab, University of Duisburg-Essen.

import cv2
import sys


def check_ForGrayImage(img):
    """ Function to decide whether the given image is gray scaled or not
        Parameters:
            img      : the input image
        Returns:
            a boolean value of either True or False
    """
    if img is None:
        sys.exit("Img is empty")
    
    # if only 2D array
    if len(img.shape) < 3:
        return True
    # if only 1 channel
    if img.shape[2] == 1:
        return True
    # if b == g and b == r than the gray values are equal in all channels
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if (b == g).all() and (b == r).all():
        return True
    return False


def make_GrayImage(img):
    """ Function to gray scaled image
        Parameters:
            img      : the input image
        Returns:
            grayscale image with one channel
    """
    
    # if only 2D array
    if len(img.shape) < 3:
        return img
    
    # if only 1 channel
    if img.shape[2] == 1:
        return img
    
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) 
    
    return -1

