#!/bin/bash

docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit \
    python detect_lanes.py \
        camera_cal \
        output_images \
        project_video.mp4 \
        processed_project_video.mp4
