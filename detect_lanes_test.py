import sys
import glob
from os import path

import matplotlib.image as mpimg
import cv2
import numpy as np

from detect_lanes import calibrate_camera
from detect_lanes import pipeline
from detect_lanes import new_filename
from detect_lanes import SRC_PERSPECTIVE_POINTS
from detect_lanes import DST_PERSPECTIVE_POINTS


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

    print("Processing test images...")

    test_image_files = glob.glob(path.join(args[2], "*.jpg"))
    for test_image_file in test_image_files:
        img = mpimg.imread(test_image_file)

        processed_image = pipeline(img, mtx, dist, M, Minv, debug_dir=args[3],
                                   original_filename=test_image_file)
        processed_image_filename = new_filename(test_image_file,
                                                args[3],
                                                "final")
        mpimg.imsave(processed_image_filename, processed_image)

    print("Done.")


if __name__ == "__main__":
    main()
