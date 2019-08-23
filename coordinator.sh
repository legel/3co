#!/bin/bash

# usage: ./coordinator number_of_gpus number_of_simulators

gpus=$1
simulators=$2
simulator=1

while [ $simulator -le $simulators ]
do
  selected_gpu=$(( $simulator % $gpus ))
  nohup bash ./render.sh $selected_gpu $simulator > $simulator.txt &
  echo "Launching simulator $simulator on GPU $selected_gpu"
  let simulator++
done
