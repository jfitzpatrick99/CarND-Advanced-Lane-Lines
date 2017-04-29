import sys

import matplotlib.image as mpimg
import cv2
import numpy as np

from detect_lanes import SRC_PERSPECTIVE_POINTS
from detect_lanes import DST_PERSPECTIVE_POINTS


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    test_image = mpimg.imread(args[0])
    cv2.polylines(test_image,
                  np.array([SRC_PERSPECTIVE_POINTS], dtype=np.int32),
                  False, [255, 0, 0], thickness=1)
    mpimg.imsave(args[1], test_image)

    cv2.polylines(test_image,
                  np.array([DST_PERSPECTIVE_POINTS], dtype=np.int32),
                  False, [0, 255, 0], thickness=1)
    mpimg.imsave(args[2], test_image)

    M = cv2.getPerspectiveTransform(np.array(SRC_PERSPECTIVE_POINTS,
                                             dtype=np.float32),
                                    np.array(DST_PERSPECTIVE_POINTS,
                                             dtype=np.float32))
    # Minv = cv2.getPerspectiveTransform(dst, src)

    imshape = test_image.shape
    img_size = (imshape[1], imshape[0])

    warped = cv2.warpPerspective(test_image,
                                 M,
                                 img_size,
                                 flags=cv2.INTER_LINEAR)
    mpimg.imsave(args[3], warped)

if __name__ == "__main__":
    main()
