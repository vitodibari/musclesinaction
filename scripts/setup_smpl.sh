#!/usr/bin/env bash

unzip ./vibe_data.zip -d ./musclesinaction/
mv ./musclesinaction/vibe_data/sample_video.mp4 .
mkdir -p $HOME/.torch/models/
mv ./musclesinaction/vibe_data/yolov3.weights $HOME/.torch/models/
