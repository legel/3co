#!/bin/bash
counter=0
while :
do
  let counter++
  echo $counter
  blender -b --python optics.py -noaudio -- 0
  aws s3 cp /home/ubuntu/research/v1 s3://3co/simulations/v1/ --recursive
  rm /home/ubuntu/research/v1/*.png
  rm /home/ubuntu/research/v1/*.json
done
