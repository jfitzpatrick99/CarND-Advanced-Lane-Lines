#!/bin/bash

docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit \
    python detect_lanes_test.py \
        camera_cal \
        output_images \
        test_images \
        output_images
