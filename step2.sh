#!/usr/bin/env sh

# Model Training with Transfer Learning
/home/kb/caffe/build/tools/caffe train --solver=/home/kb/airplanes_caffe/AlexNet/solver.prototxt --weights /home/kb/airplanes_caffe/AlexNet/bvlc_reference_caffenet.caffemodel 2>&1 | tee /home/kb/airplanes_caffe/AlexNet/alexnet.log

echo "done"
