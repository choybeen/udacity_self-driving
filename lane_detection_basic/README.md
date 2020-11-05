# lane_detection
 the nanodegree projects for self-driving

## Step1)  first download the training data 

## step2, thread color image only keep yellow and white

## step3, from yellow part to gray image, possible lane areas

## step4, blur image with filter to remove noises, and detect edge by canny

 ![GitHub Logo](https://github.com/choybeen/udacity_self-driving/blob/main/lane_detection_basic/Capture.JPG?raw=true)
 
## step5, remove other areas by ROI, keep only front area before car

## step6, apply the Hough Transform

## step7, mark the left and right lane area seperately

## step8, mark the left and right lane area seperately

 ![GitHub Logo](https://github.com/choybeen/udacity_self-driving/blob/main/lane_detection_basic/Capturer.JPG?raw=true)
 
## step9, mark the lane area by left/right line

# this process methond could applied on some videos

since the secereo change quick in reality, but the lane position and angle values are continue, tracking the lane of next frame in near area avoid jittering 

 [![Fibonacci RMI Java EE](https://github.com/choybeen/udacity_self-driving/blob/main/lane_detection_basic/Capturev.JPG?raw=true)](https://youtu.be/9D5ahYHA_nE)

 [![Fibonacci RMI Java EE](https://github.com/choybeen/udacity_self-driving/blob/main/lane_detection_basic/Capturev2.JPG?raw=true)](https://youtu.be/naG1hxNbbY4)



