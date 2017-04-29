# detect_lanes.py - Detect lanes in a video input stream and produce an output
# video stream showing the lane lines.
import matplotlib
matplotlib.use("Agg")

import sys
import glob
from os import path

from moviepy.editor import VideoFileClip
import matplotlib.image as mpimg
import numpy as np
import cv2


SRC_PERSPECTIVE_POINTS = [[190,  720],
                          [617,  435],
                          [663,  435],
                          [1120, 720]]


DST_PERSPECTIVE_POINTS = [[325, 720],
                          [325, 0],
                          [955, 0],
                          [955, 720]]


def new_filename(filename, new_dir, file_qualifier):
    """
    Returns a new filename based on the provided filename but in a new directory
    incorporating the given filename qualifier.

    `filename` Filename to transform
    `new_dir` Directory to create the new filename for
    `file_qualifier` String that describes something about how the file has been
    transformed

    Given a filename "foo/bar.jpg" and that the "new_dir" parameter has a value
    of "baz" and the "file_qualifier" is debug, this function returns a filename
    of "baz/bar-debug.jpg".
    """
    basename = path.basename(filename)
    file_and_ext = basename.split(".")
    result = path.join(new_dir,
                       file_and_ext[0] +
                       "-" + file_qualifier + "." + file_and_ext[1])
    return result


def output_debug_image(debug_dir, original_filename, file_qualifier, img,
                       cmap=None):
    """
    Output an image for debugging purposes if debug_dir is not None.

    `debug_dir` is the directory to write images to for debugging purposes
    `original_filename` specifies the original filename of the image being
    processed
    `file_qualifier` is a tag to attach to the image
    `img` is the image to write
    `cmap` is the name of the color map to use when saving the image
    """
    if debug_dir:
        filename = new_filename(original_filename,
                                debug_dir,
                                file_qualifier)

        if cmap:
            mpimg.imsave(filename, img, cmap=cmap)
        else:
            mpimg.imsave(filename, img)


def calibrate_camera(cal_dir, undistorted_dir, glob_pattern="*.jpg",
                     checker_board_shape=(9, 6)):
    """
    Calibrates the camera using the chessboard images contained in the given
    folder.

    `cal_dir` Specifies the directory containing the images to use for
    calibration.
    `undistorted_dir` Directory to use to output undistorted calibration images.
    `glob_pattern` Specifies a glob pattern to use to find images to use for
    calibration.
    `checker_board_shape` specifies the shape of the checker board used in the
    calibration.

    This function returns the camera matrix and the distortion coefficients in
    that order.

    Note that this function borrows heavily from the video in part 10 of the
    project lesson.
    """
    # List of 2D points in the image plane.
    img_points = []
    # List of 3D points in real world space.
    obj_points = []
    images = []

    # Prepare object points of the form (0, 0, 0)...(8, 5, 0)
    objp = np.zeros((checker_board_shape[0] * checker_board_shape[1], 3),
                    np.float32)
    objp[:, :2] = np.mgrid[0:checker_board_shape[0],
                           0:checker_board_shape[1]].T.reshape(-1, 2)

    cal_shape = None
    cal_image_files = glob.glob(path.join(cal_dir, glob_pattern))

    for cal_image_file in cal_image_files:
        # Read image file.
        img = mpimg.imread(cal_image_file)

        # Convert image to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cal_shape = gray.shape[::-1]

        # Try to find chessboard corners in the image.
        ret, corners = cv2.findChessboardCorners(gray,
                                                 checker_board_shape,
                                                 None)

        if ret:
            # Chessboard corners found; update the lists
            img_points.append(corners)
            obj_points.append(objp)

            img = cv2.drawChessboardCorners(img,
                                            checker_board_shape,
                                            corners,
                                            ret)
            images.append((cal_image_file, img))
        else:
            print("[WARNING] Chessboard corners not found for image "
                  + cal_image_file)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                       img_points,
                                                       cal_shape,
                                                       None,
                                                       None)

    # Undistort the images and write them out to directory for undistorted
    # images.
    for image in images:
        dst = cv2.undistort(image[1], mtx, dist, None, mtx)

        undistorted_image_file = new_filename(image[0],
                                              undistorted_dir,
                                              "undistorted")
        mpimg.imsave(undistorted_image_file, dst)

    return mtx, dist


def to_single_channel(img, color_space="HLS", channel="S"):
    """
    Convert the image to a representation in a single channel.

    `img` is the image in RGB format
    `color_space` is the destination color space for the image
    `channel` is the channel to extract from the image

    If "color_space" is set to "GRAY" then the image is converted to grayscale.
    and the "channel" parameter is ignored.
    """
    if color_space == "GRAY":
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if color_space == "HLS":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        channel_indices = {"H": 0, "L": 1, "S": 2}
        return img[:, :, channel_indices[channel]]

    if color_space == "RGB":
        channel_indices = {"R": 0, "G": 1, "B": 2}
        return img[:, :, channel_indices[channel]]


def abs_sobel_thresh(img, color_space="HLS", channel="S", orient='x',
                     sobel_kernel=3, thresh=(20, 100)):
    """
    Apply the Sobel operator to find the derivative in the given orientation.

    `img` is the undistorted image in RGB format
    `orient` specifies the direction to calculate the gradient in
    `sobel_kernel` specifies the kernel size
    `thresh` is the minimum and maximum gradient values to keep
    """
    img = to_single_channel(img, color_space=color_space, channel=channel)
    if orient == "x":
        sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    # Absolute derivative to accentuate lines away from given orientation
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Threshold gradient
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel >= thresh[0]) &
                 (scaled_sobel <= thresh[1])] = 1
    return sobel_binary


def color_thresh(img, color_space="HLS", channel="S", thresh=(170, 255)):
    """
    Return a binary image where pixels with the given threshold are replaced
    with 1's and all other pixels are 0.

    `img` is the undistorted image in RGB format
    `color_space` is the format to use for the thresholding and can be one of
    HLS or RGB
    `channel` is the channel to do the thresholding on
    `thresh` is the threshold to use
    """
    img = to_single_channel(img, color_space=color_space, channel=channel)
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) &
           (img <= thresh[1])] = 1

    return binary


def mag_thresh(img, color_space="HLS", channel="S", sobel_kernel=3,
               thresh=(30, 100)):
    """
    Return a binary image where the magnitude of the sobel gradient is within
    the given theshold.

    `img` is the undistorted image in RGB format
    `color_space` is the format to use for the thresholding and can be one of
    HLS or RGB
    `channel` is the channel to do the thresholding on
    `sobel_kernel` is the size of the sobel kernel to use
    `thresh` is the threshold values to use
    """
    img = to_single_channel(img, color_space=color_space, channel=channel)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_xy = np.sqrt(sobel_x**2 + sobel_y**2)
    scaled_sobel = np.uint8(255 * abs_sobel_xy / np.max(abs_sobel_xy))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1

    return binary_output


def dir_threshold(img, color_space="HLS", channel="S", sobel_kernel=3,
                  thresh=(0.7, 1.3)):
    """
    Return a binary image that applies a threshold based on the direction of
    the gradient in a given image.

    `img` is the undistorted image in RGB format
    `color_space` is the format to use for the thresholding and can be one of
    HLS or RGB
    `channel` is the channel to do the thresholding on
    `sobel_kernel` is the size of the sobel kernel to use
    `thresh` is the threshold values to use
    """
    img = to_single_channel(img, color_space=color_space, channel=channel)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    arctan_sobel = np.arctan2(abs_sobel_y, abs_sobel_x)
    binary_output = np.zeros_like(arctan_sobel)
    binary_output[(arctan_sobel >= thresh[0]) & (arctan_sobel <= thresh[1])] = 1

    return binary_output


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)

    # Define a 3 channel or 1 channel color to fill the mask with depending
    # on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # Fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # Return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def find_lanes(binary_warped, nwindows=9, visualize=False):
    """
    Find the lanes in the binary warped version of the image using the histogram
    and sliding windows technique.

    `binary_warped` is the warped binary version of the image
    `nwindows` is the number of windows to use
    `visualize` is a boolean value indicating if the detected lane lines should
    be visualized and returned.

    This function returns the 2nd order polynomial coefficients of the line that
    best fits each lane line as well as the visualized image if requested.

    Note that this function is almost verbatim from lesson 10 of the course.
    """
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = None
    if visualize:
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 100

    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        if visualize:
            cv2.rectangle(out_img,
                          (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img,
                          (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) &
                          (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) &
                           (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean
        # position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if visualize:
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = \
            [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = \
            [0, 0, 255]
        # plt.imshow(out_img)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)

    return left_fit, right_fit, out_img


def draw_lanes(img, warped, Minv, ploty, left_fitx, right_fitx):
    """
    This function draws the lanes on the given image.

    `img` is the image to draw the lanes on
    `warped` is the binary warped image
    `Minv` is the inverse perspective transform matrix to transform warped back
    to the original image perspective.
    `ploty` are the y points for the lane lines
    `left_fitx` are the x coordinates of the left lane line
    `right_fitx` are the x coordinates of the right lane line
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    # matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp,
                                  Minv, (img.shape[1], img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    return result


def calc_radius(ploty, left_fitx, right_fitx, ym_per_pix=30/720,
                xm_per_pix=3.7/700):
    """
    Calculate the radius of the lane lines given their coordinates

    `ploty` are the y coordinates of the lane lines
    `left_fitx` are the x coordinates of the left lane line
    `right_fitx` are the x coordinates of the right lane line
    `ym_per_pix` is the ratio of meters per pixel in the y direction
    `xm_per_pix` is the ratio of meters per pixel in the x direction
    """
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    y_eval = np.max(ploty)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix +
                     left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                      right_fit_cr[1])**2)**1.5) / \
        np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad


def calc_lane_position(imshape, ploty, left_fitx, right_fitx,
                       xm_per_pix=3.7/700):
    """
    Calculate the position of the vehicle in the lane.

    `imsahpe` is a tuple containing the dimensions of the image
    `ploty` are the y coordinates of the lane lines
    `left_fitx` are the x coordinates of the left lane
    `right_fitx` are the x coordinates of the right lane
    `xm_per_pix` is the ratio of meters per pixel in the x direction
    """
    vehicle_center_x = int(imshape[1] / 2)
    lane_left_x = left_fitx[len(left_fitx) - 1]
    lane_right_x = right_fitx[len(right_fitx) - 1]

    lane_width = lane_right_x - lane_left_x
    lane_center_x = (lane_width / 2) + lane_left_x

    delta_vehicle_x = vehicle_center_x - lane_center_x

    return delta_vehicle_x * xm_per_pix


def pipeline(img, mtx, dist, M, Minv,
             debug_dir=None, original_filename=None):
    img = cv2.undistort(img, mtx, dist)
    img = np.copy(img)

    imshape = img.shape

    # Sobel x
    sx_binary = abs_sobel_thresh(img, color_space="HLS", channel="S",
                                 thresh=(20, 100))
    output_debug_image(debug_dir, original_filename, "sobel_x", sx_binary,
                       "gray")

    # Sobel y
    sy_binary = abs_sobel_thresh(img, color_space="HLS", channel="S",
                                 orient="y", thresh=(20, 100))
    output_debug_image(debug_dir, original_filename, "sobel_y", sy_binary,
                       "gray")

    # Threshold S color channel in HLS space
    s_binary = color_thresh(img, color_space="HLS", channel="S",
                            thresh=(90, 255))
    output_debug_image(debug_dir, original_filename, "s_thresh", s_binary,
                       "gray")

    mag_binary = mag_thresh(img, thresh=(30, 100))
    output_debug_image(debug_dir, original_filename, "mag_thresh", mag_binary,
                       "gray")

    # Directional gradient
    dir_binary = dir_threshold(img, thresh=(0.7, 1.3))
    output_debug_image(debug_dir, original_filename, "dir_grad", dir_binary,
                       "gray")

    # Combine the binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[((sx_binary == 1) & (sy_binary == 1)) |
                    ((mag_binary == 1) & (dir_binary == 1) &
                     (s_binary == 1))] = 1
    output_debug_image(debug_dir, original_filename, "final_binary",
                       combined_binary, "gray")

    # Do perspective transform
    warped = cv2.warpPerspective(combined_binary,
                                 M,
                                 (imshape[1], imshape[0]),
                                 flags=cv2.INTER_LINEAR)
    output_debug_image(debug_dir, original_filename, "warped",
                       warped, "gray")

    left_fit, right_fit, out_img = find_lanes(warped, visualize=True)
    output_debug_image(debug_dir, original_filename, "lane_fit",
                       out_img)

    # Generate x and y values for plotting
    ploty = np.linspace(0,
                        warped.shape[0] - 1,
                        warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    img = draw_lanes(img, warped, Minv, ploty, left_fitx, right_fitx)

    left_curverad, right_curverad = calc_radius(ploty, left_fitx, right_fitx)

    curverad = (left_curverad + right_curverad) / 2.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,
                "Radius of lane curvature is: {0:.2f} meters".format(curverad),
                (10, 30), font, 1, (255, 255, 255), 2)

    lane_pos = calc_lane_position(img.shape, ploty, left_fitx, right_fitx)

    if abs(lane_pos) <= 0.1:
        cv2.putText(img,
                    "Vehicle is in the center of the lane".format(lane_pos),
                    (10, 60), font, 1, (255, 255, 255), 2)
    elif lane_pos > 0.0:
        cv2.putText(img,
                    "Vehicle is {0:.2f} meters right of center"
                    .format(lane_pos),
                    (10, 60), font, 1, (255, 255, 255), 2)
    else:
        cv2.putText(img,
                    "Vehicle is {0:.2f} meters left of center"
                    .format(abs(lane_pos)),
                    (10, 60), font, 1, (255, 255, 255), 2)

    return img


#
# Program entry point.
#
def main(args=None):
    if args is None:
        args = sys.argv[1:]

    print("Calibrating camera...")

    mtx, dist = calibrate_camera(args[0], args[1])

    print(mtx)
    print(dist)

    print("Camera calibrated.")

    print("Calculating perspective transform...")

    M = cv2.getPerspectiveTransform(np.array(SRC_PERSPECTIVE_POINTS,
                                             dtype=np.float32),
                                    np.array(DST_PERSPECTIVE_POINTS,
                                             dtype=np.float32))
    Minv = cv2.getPerspectiveTransform(np.array(DST_PERSPECTIVE_POINTS,
                                                dtype=np.float32),
                                       np.array(SRC_PERSPECTIVE_POINTS,
                                                dtype=np.float32))

    print("Perspective transform calculated.")

    print("Processing video...")

    clip = VideoFileClip(args[2])

    def parameterized_pipeline(img):
        return pipeline(img, mtx, dist, M, Minv)

    processed_clip = clip.fl_image(parameterized_pipeline)
    processed_clip.write_videofile(args[3], audio=False)

    print("Done processing video.")


if __name__ == "__main__":
    main()
