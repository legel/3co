#!/bin/bash

rm x.mp4
rm output.mp4
ffmpeg -y -start_number 0 -i scipy_optimization_outputs/%d.png x.mp4
ffmpeg -y -i x.mp4 -pix_fmt yuv420p output.mp4
