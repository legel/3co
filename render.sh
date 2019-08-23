#!/bin/bash

gpu=$1
simulator=$2
directory=sim${simulator}_gpu${gpu}

mkdir -p /home/ubuntu/research/$directory

counter=0
while :
do
  echo $counter
  blender -b --python optics.py -noaudio -- $gpu $directory
  aws s3 cp /home/ubuntu/research/$directory s3://3co/simulations/v1/ --recursive
  rm /home/ubuntu/research/$directory/*.png
  rm /home/ubuntu/research/$directory/*.json
  let counter++
done
