**This project was concluded during my occupation in the Language Technology Lab department at the University of Duisburg-Essen. My thanks go to all the researchers and staff who have helped me make this project to a reality and to the department for allowing me to make the project public.**

# Ruling Line Removal using image progessing

The performance of any Optical Character Recognition (OCR) heavily depends on the purity of the input image. It is however often the case that these images contain noise or/and background lines which are known as "Ruling-Lines". These lines are used to assist writing of any kind such as music notes and textbooks. Since they are used to guide the writer, the lines and the writings cross each other on multiple occasions. Hence, it is imperative for these lines to be detected and removed prior to the OCR. The detection and removal of the "Ruling-Lines" have however many obstacles, usually caused by scanning, such as: broken lines, crooked or non-horizontal lines, crossing with additional printing. 

This project presents a straightforward solution to the detection and removal of the Ruling-Lines. In the section below, each step of the algorithm is explained in detail with addtional imagery to help better picture what each step accomplishes.

# Explanation of the Code

The ruling line removal was performed using the following four main steps:

1. Preprocessing
2. Calculating the Line Distance
3. Finding the Ruling Lines
4. Removing the Ruling Lines

Below is a sample text image that will be used to explain what each of the algorithm actually accomplishes:

![Sample_text](/doc/sample-1.png)

## Preprocessing

The dataset for this project consists of grey scaled images each of which contain two pages of writing and sometimes additional unwanted printing such as the line numbers at the start of each line and/or the company logo. Furthermore the position of the pages occasionally shift, due to not equal scaning. Hence, a preprocessing step is required where the image is split into two images (one for each page) such that each image contains only the ruling lines and/or text, and nothing else. 

This is, in general, achieved by using the four black squares located at four sides (upper-left, upper-right, below-left and below-right) of each image as anchor points. More precisly, the following steps were conducted:

1.	For cropping the left image, the left-most 10% of the image is considered. 
2.  All the contours within this image are found. 
3.	For each found contour, we form a bounding rectangle. We decide then that we have found our two black squares (upper-left and below-left), only if the area of the bounding rectangle is equal to the heurestically calculated area of the black square ± some deviation. 
4.	The image is then cropped out using the y-coordinate of the upper-right corners of the black squares the heurestically calculated width of the page. 
5.  All the previous steps are then once more executed for the right page.

The advantage of this approach is that even if the whole image shifts, the position of the ruling lines with respect to the black square does not. 

Below is a small demonstration of how the algorithm works for the left page. The full lines correspond to the y-coordinates from the black squares while the interrupted line corresponds to the width of the page which is to be heurestically calculated such that no text is damaged:

![preprocessing_demo](/doc/preprocessing_demo.png)

The preprocessing step produces the following two images:

Left Page                  |  Right Page
:-------------------------:|:-------------------------:
![sample-1_lp](/doc/sample-1_lp.png)  |  ![sample-1_rp](/doc/sample-1_rp.png)
 
## Calculating the Line Distance

Now that we have a nicely cropped out page, we will move to the first step in detecting the Ruling-Lines: calculating the line distance. It should be noted that the approach proposed in the 2015 paper by Mohammed A. A. Rafaey “Ruled lines detection and removal in grey level handwritten image documents” was implemented with some adjustments. The steps are as follows:

1. The page is first divided into square-shaped windows where each side is 1/5th of the width of the page. The division is started from the bottom and the unfinished squares at the top are neglected. The windows are numerated as Wij where i represents the row and j represents the column of the window. 
2.	Then the windows W32, W34, W52 and W54 are cropped out. 
3.	For each window Hough transformation is applied to find two of the strongest detected lines in each window and the end points of these lines are stored. 
4.	Finally the difference between the y-coordinate of the two strongest lines (if necessary normalized with the number of ruling lines in between) are calculated for each window and the median of these values is taken as the line distance for the page.

Below is W52 of the left page, detected lines in W52 (threshold for Hough transfrom set to one third of width of W52) and strongest two lines respectively:

W52 - Left page            |  W52 - detected lines(in red)     | W52 - strongest two lines(in green)
:-------------------------:|:-------------------------:|:-------------------------:
![W52_lp](/doc/W52.png)  |  ![detected_W52](/doc/detected_W52.png)  |  ![strongest_2_W52](/doc/strongest_2_W52.png)

##	Finding the Ruling Lines

To find the ruling lines, instead of examinig the whole image, the best representative start and end point of each of the ruling line was searched. It should be noted that the actual ruling lines have a certain width, however due to the low quality of the scans, most ruling lines were discontinuous. That's why it was more sensible to find the “best representatives”. Below are the following steps:

1.	Firstly, 25% of the image from the left is cropped. This initial cropping ensures that the discontinuity of the ruling lines does not have an effect on our method. 
2.	Then the horizontal histogram profile for the image is made and the one dimensional signal is smoothed out with Gaussian Blur with a kernel size of 51. After that the maxima of the smoothed signal is determined. 
3.	The final step is finding local maximum points that are apart from one another as much as the line distance and searching around these points in the orginal signal to find the real position of the ruling lines. 
4. All the previous steps are then once more executed to find the end points of the ruling lines. 

It was seen that there was an average of 2-3px difference between the starting and ending point of the ruling lines with the starting point being usually at a higher position.

Below are two images to better demonstrate what this step accomplishes using left page of the sample-1.png as an example. See how in the left image, the green lines passes diretcly through the middle of the ruling lines in the left half of the images, and how the right image achieves the same for the right half of the image's ruling lines.

Left Page - Start Points            |  Left Page - End Points      
:-------------------------:|:-------------------------:
![sample-1_lp_start](/doc/sample-1_lp_start_on_binary_image.png)  |  ![sample-1_rp_start](/doc/sample-1_lp_end_on_binary_image.png)  

## Removing the Ruling Lines

The first and foremost thing to determine for any ruling line removal algorithm is to tell the algorithm where the ruling lines are (i.e. define a "window" for the removal algorithm to look at). So far, we have y-coordinate pairs for the (best representative) start and end points of each ruling line. These start and end points usually denote the middle of the ruling lines, however it is often the case that there are also pixels above and below these start and end points. Hence, it makes sense have parameters to further help us indicate where the ruling lines are approximately located at. These parameters in the above code are called window start(window_start) and window end(window_end):

* window_start: denotes the higher limit for the window
* window_end: denotes the lower limit for the window

Below is a small visualization to explain the above mentioned parameters. Assuming the black line is our ruling line, and the two green lines denote the start and end points of our ruling line (calculated in the previous step), one can easily see that there are still many pixels below start point and many above the end point (which is usually also the case in real life). The parameters window_start and window_end are hence needed to be adjusted so that an area is specified which covers all of the ruling line:

![removal_parameters](/doc/removal_parameters.png)

One last parameter which is also crucial is ruling line width(line_thickness).:

* ruling line width: is the expected width of the ruling lines. The parameter is used by the "majority voting" and "line iterator" algorithm to decide whether a block of pixels belong to handwriting or ruling line.

For removing the ruling lines the following three algorithms were used:

### Brutal Algorithm: 
The brutal algorithm changes every pixel in the area specified by the ruling line width, window start and window end parameters into background color. The algorithm does not/ist not able to differentiat between text and ruling line, hence the name "brutal". Inefficient as it may be, brutal algorithm achieves the highest score when it comes to removing the ruling lines (but of course does it at the cost of removing handwriting).

### Majority Voting: 
The majority voting, just like in brutal algorithm, considers every parameter in the area by the ruling line width, window start and window end parameters into background color. Then for each row of this area, the number of black pixels are counted. If the number of black pixels are greater than the line thickness, then it is assumed that it belongs to the text and this row remains as it is. However, if the number of black pixels are less than or equal to the line thickness, this row is turned to the background color.
  
      With majority voting, we introduce a way to differentiate the pixels belonging to the ruling line and to the text.

### Line Iterator: 
For the line iterator, first a line is drawn for each ruling line that connects the ruling line's start and end point.  After that the algorithm visits each pixel along the line and for each pixel visited a row is formed specified by the ruling line width, window start and window end parameters. Then we count the number of black pixels in this row and if it is less than our equal to the line thickness, we assume it belongs to the ruling line and turn it into the background color, if not, we leave it as it is.

      The idea with this method is that the line that we draw at the beginning acts as a best fitting line to our often separated ruling lines, ultimately producing a better answer.

# Evaluation

In the evaluation step, the performance of all three algorithms are compared.

The first step in evaluation is to manually remove the ruling line of the first 7 lines of a page and then compare the original image with the new image without ruling lines. The right page of the sample with manually removed ruling lines is below:

![original_page_no_line.png](/doc/original_page_no_line.png)

The evaluation step then performs two operations:

1. Construct a three channel image where green denotes correctly removed pixels, red denotes incorrectly removed pixels and blue denotes pixels that should have been removed but were not. 

Below is the image produced by the evaluation function when using Line Iterator algorithm with the parameters: line_width = 5, window_start = 2 and window_end = 6

![original_page_no_line.png](/doc/sample-1_rp_LineIterator_eval.png)

2. Print the statistics related to evaluation such as "correctly removed pixels" and "correctly removed ruling line pixels". Below is a table containing the best statistics achieved for every removal algorithm using the image without ruling lines above along with their parameters (written in the following order: line_width, window_start, window_end):


|Searched Parameters                    |Brutal (5, 5, 7)   | Majority Voting (5, 5, 7) | Line Iterator (5, 2, 6)|
|      :------:                         |:----------------: |:----------------------:   |:--------------------:  |
|Number of pixels removed               | 46219             | 36766                     | 39210                  |
|Correctly Removed Ruling Line Pixels (%)   | 99.60             | 93.53                     | 96.50                  |
|Correctly Removed Pixels (%)               | 78.16             | 92.27                     | 89.26                  |

    It should be noted that the above results are only from the sample page. In the actual dataset of the project, the performance of each algorithm was a bit underperforming compared to the above results. 

# Conclusion and Final Remarks

If we were to compare the removal algorithms above:

1. Brutal Algorithm removes almost every pixel belonging to the ruling line, but does it at the cost of handwriting as 21.84% of the total removed pixels belonged to the handwriting.
2. Majority Voting Algorithm provides a more balanced pair of statistics and, as expected, is more reliable compared to the brutal algorithm.

*It should be noted that since both brutal and majority voting algorithm are built upon the same idea, which is working in the rectangular area specified by the three parameters, their "best case" parameters are in most cases similar or like in this example the same.*

3. Line Iterator Algorithm's parameter choice differs from the rest as the algorithm moves scanns the ruling line in a different manner. The result of this algorithm heavily depends on the parameters, meaning even the slightest change can cause major changes. For example, if one were to change the second parameter from 2 to 3, we would have 92.94% for "Correctly Removed Ruling Line Pixels" and 94.11% for "Correctly Removed Pixels", a result which can also be considered the "best" depending on the situation.

Compared to Majority Voting algorithm, Line Iterator performs either better on removing the ruling line pixels or correctly removing the pixels, depending on the parameters chosen, but suprisingly never both. 

Consequently, if one wishes to strictly have a statistics at best, while also minizing the loss on the other, Line Iterator is a suitable algorithm. If an overall balanced results are desired than Majority Voting should be chosen. Finally, if one is lucky enough to have a dataset where the ruling lines are all horizontal and are completely sperated with handwriting, one might consider using the Brutal algoritm.

Finally, the results of the removal algorithms may not be state of the art, but nonetheless provide a straightforward and to some extent flexible answer to the important Ruling Line Removal problem that is present today.
