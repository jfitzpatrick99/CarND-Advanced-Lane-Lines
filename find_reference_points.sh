#!/bin/bash

docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit \
    python find_reference_points.py \
        test_images/straight_lines1.jpg \
        output_images/straight_lines1-src_ref_lines.jpg \
        output_images/straight_lines1-dst_ref_lines.jpg \
        output_images/straight_lines1-warped.jpg
