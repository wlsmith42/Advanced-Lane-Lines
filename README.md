# **Advanced Finding Lane Lines on the Road**
---

**Advanced Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[image1]: ./test_images/test5.jpg "image"
[image2]: ./output_images/test5_undistorted.jpg "undist"
[image3]: ./output_images/test5_binary.jpg "binary"
[image4]: ./output_images/test5_perspective_transform.jpg "warp"
[image5]: ./output_images/test5_fitted_lines.jpg "fitted"
[image6]: ./output_images/test_images/test5.jpg "result"

[image7]: ./camera_cal/calibration2.jpg "cali"
[image8]: ./output_images/calibration2_undistorted.jpg "cali-undist"


---

## Reflection

### Pipeline Description

My pipeline consisted of the following steps:
* Correct image distortion
* Apply Gaussian smoothing
* Create a thresholded binary image
* Apply a region mask to the image
* Perspective transform the image
* Identify lane line pixels and calculate the lane's curvature
* Visualize lane and warp the rendering onto the original image
* Output curvature and distance from center onto the image

Before passing images through the pipeline, there is some pre-processing to do. To accomplish this, I used a series of 20 images of a chessboard taken at various angles. Using openCV's findChessboardCorners, I can easily find the corners and add the returned items to an objectpoints and imagepoints array to store the 3d points in real world space and the 2d points in the image plane respectively. The image on the left shows the raw image while the image on the right has been undistorted.
![alt text][image7] ![alt text][image8]


Now the objectpoints and imagepoints can be applied to the actual image (on the left) and returns the undistorted image (on the right).
![alt text][image1] ![alt text][image2]


Next I apply Gaussian smoothing to help with the thresholding in the next step. This ensures that the gradients are smoothed out and makes it easier to distinguish lines.

Creating the Binary threshold requires color transforms and gradients. For color transforms I used RGB, HLS, HSV, and LAB color spaces. After trying several low/high thresholds for each channel in the four color spaces, I ended up using the B, S, V, and L channels from each color space. For each of these channels I calculated the Sobel and the X-gradient to detect edges and emphasize the near vertical edges.
```Python
def sobelThresh(img_channel, s_thresh, sx_thresh):
    """
    Calculates gradient of image to be used to detect edges.
    """
    #Apply x gradient to emphasize verticle lines
    sobelx = np.absolute(cv2.Sobel(img_channel, cv2.CV_64F, 1, 0))
    #Resize to an 8-bit integer
    scaled_sobel = np.uint8(255*sobelx/np.max(sobelx))

    #Create binary image based on X-Gradient
    sobel = np.zeros_like(scaled_sobel)
    sobel[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    #Create binary image based on Sobel only
    chanThresh = np.zeros_like(img_channel)
    chanThresh[(img_channel >= s_thresh[0]) & (img_channel <= s_thresh[1])] = 1

    #Combine x-gradient and sobel binary image
    combined = np.zeros_like(chanThresh)
    combined[(chanThresh == 1) | (sobel == 1)] = 1

    return combined


def threshold(img, s_thresh=(170,255), sx_thresh=(20,100)):
    """
    Creates a thresholded binary image based on color transforms
    and gradients.
    """

    #RGB color space
    B_channel = img[:,:,2] #Clear lines on some images, low-moderate noise
    B_channel = sobelThresh(B_channel, (220,255), (20,100))

    #HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S_channel = hls[:,:,2] #Clear lines on all images, low noise
    S_channel = sobelThresh(S_channel, s_thresh, sx_thresh)

    #HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2] #Clear lines on most images, low-moderate noise
    v_channel = sobelThresh(v_channel, (220,255), (20,100))

    #LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel = lab[:,:,0] #Clear lines on most images, low-moderate noise
    l_channel = sobelThresh(l_channel, (210,255), (20,100))

    #Combine the best channels from each color space
    thresh = np.zeros_like(S_channel)
    thresh[(S_channel == 1) | (B_channel == 1) | (v_channel == 1) | (l_channel == 1)] = 1

    return thresh
```
When the Sobel and X-gradient are applied to the B, S, V, and L channels the resulting binary threshold is shown in the image below.
![alt text][image3]


After the binary image is achieved, I apply a region mask to isolate the lane lines. This reduces the noise from around and above the lines to help improve the performance of the line detection later on.

Next is the perspective transform. This involves setting four points on the image roughly outlining the lane and transforming them into parallel lines that represent a bird's eye view of the lane. openCV's getPerspectiveTransform takes the source and destination points and returns the transform matrix that is needed to created the warped lane.
```Python
def perspectiveTransform(img):
    """
    Warps an image to create a bird's eye view of the lane for use
    in calculating the curvature of the lane.
    """

    #Get image data
    max_y = img.shape[0]
    max_x = img.shape[1]
    img_size = (max_x, max_y)

    #Create source image points
    src = np.float32([[max_x*0.465, max_y*0.625], #top left
                     [max_x*0.156, max_y],      #bottom left
                     [max_x*0.879, max_y],     #bottom right
                     [max_x*0.539, max_y*0.625]]) #top right

    #Create destination image points
    dst = np.float32([[max_x*0.234,0],
                     [max_x*0.234, max_y],
                     [max_x*0.742, max_y],
                     [max_x*0.742, 0]])

    #Get matrix to show how to transform image from source to destination points
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    #Warp the image
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M, Minv
```
The following image is an example of a warped lane.
![alt text][image4]


At last we are able to fit lines to the image. The code below is for a image with no prior line data, but this project is equipped to compute a weighted average on prior line data. The lines are detected using a histogram to find the highest pixel values in the image. Then that x-position is used as a sliding window to find other high pixel areas within a margin of the previously detected lane. Numpy's polyfit function is used to fit polynomials with the selected x-values.
```Python
def find_lane_pixels(binary_warped, Lline, Rline):
    """
    Fit lines to left and right lanes when there is no
    previous lane data
    """

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 20
    # Set the width of the windows +/- margin
    margin = 50
    # Set minimum number of pixels found to recenter window
    minpix = 75

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        #Find the four below boundaries of the window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2)

        #Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    #Create fitted lines
    Lfit = np.polyfit(lefty, leftx, 2)
    Rfit = np.polyfit(righty, rightx, 2)

    #Add fitted lines to left and right lines
    Lline.addFit(Lfit)
    Rline.addFit(Rfit)

    return Lfit, Rfit
```
After the lines have been detected, they are drawn on the warped image as shown below.
![alt text][image5]

The next step is to calculate the curvature of the lane as well as the distance of the vehicle from center. To calculate the lane curvature, I used the equation: \(\[R_{Curve} = \frac{(1 + (2Ay + B)^{2})^{\frac{3}{2}}}{\left | 2A\right |}\]\)
To calculate the distance from center I simply subtracted the center of the lane from the center of the image (which represented the middle of the car) and converted from pixels to meters.
```Python
#Convert pixels to meters
ym_per_pix = 30/720
xm_per_pix = 3.7/650

#Calculate fits with real-world measurements
left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

#Set real-world y value
y_eval = np.max(lefty) * ym_per_pix

#Calculate curve values for each line
left_curve =  ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curve =  ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
#Round curve value to the nearest integer
left_curve = int(round(left_curve))
right_curve = int(round(right_curve))

#if both lines fitted, calculate distance from lane center
if left_fitx is not None and right_fitx is not None:
    #True center of image - represents where the car is
    img_center = img_shape[0] * xm_per_pix
    #Real world integer value of the lines location
    Lfit_int = left_fit_cr[0]*y_eval**2 + left_fit_cr[1]*y_eval + left_fit_cr[2]
    Rfit_int = right_fit_cr[0]*y_eval**2 + right_fit_cr[1]*y_eval + right_fit_cr[2]
    #Shows where the line center is in the image
    car_position = (Rfit_int + Lfit_int) / 2
    #Determine distance from the center of the frame to the center of the lane
    center_dist = (img_center - car_position)
    #Round answer to two decimal places
    center_dist = round(center_dist, 2)
```


Finally, after all the steps are put together the result is shown in the image below.
![alt text][image6]





### Shortcomings of the Pipeline

The two biggest shortcoming of the pipeline come from steep curves, extraneous lines on the pavement, and direct sunlight. This pipeline can't detect lanes with steep lines or in direct sunlight. Extraneous lines, like the construction in challenge_video.mp4, can be avoided for the most part. I made countless versions of the pipeline; some performed better with extra lines, other with direct sunlight. In the end I went with the one that covered all cases evenly because a specialized lane finding algorithm won't be of much use in the real world.




### Improvements to the Pipeline

To detect steep curves, I would need to implement another solution besides this sliding window approach. This is because steep curves move one lane into the detection area of the other lane, causing the algorithm to believe it is still a semi-straight lane. One potential solution could be applying a vertical sliding window in addition to the horizontal window, although that may prove not worth the extra computation.

The direct sunlight problem is more more difficult. Since the bright light washes out the colors, no amount of thresholding can retrieve data from all-white pixels. Instead, the solution for this could be to create a line planning algorithm that will predict where the line will go next. This would be difficult to get accurate enough to work in the steep curves of the harder_challenge_video.mp4, but I believe it would be a better and more robust solution that relying entirely on getting clear data from the on-board cameras.
