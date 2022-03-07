#!/bin/bash

# clean up
rm intermediate_output.mp4
rm video.mp4
rm reversed.mp4
rm final.mp4

# get all images with .png name and save
ffmpeg -y -start_number 0 -i artcloud_gv004_%04d.png -filter:v fps=24 intermediate_output.mp4

# re-encode for Mac, mainstream viewers
ffmpeg -y -i intermediate_output.mp4 -pix_fmt yuv420p video.mp4

# # make a reversed video of video.mp4
# ffmpeg -i video.mp4 -vf reverse reversed.mp4

# # combine videos into final output
# ffmpeg -f concat -safe 0 -i mylist.txt -c copy final.mp4
