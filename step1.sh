#!/usr/bin/env sh

CALDIR="/home/kb/airplanes_caffe"
CAFFE_ROOT="/home/kb/caffe"

echo "create train, val"
python $CALDIR/codes/create_train_val.py

echo "create lmdb"
python $CALDIR/codes/create_lmdb.py 

echo "computing mean of images"
# compute mean
$CAFFE_ROOT/build/tools/compute_image_mean $CALDIR/data/lmdb/train_lmdb $CALDIR/data/lmdb/mean.binaryproto

echo "create png graphic of the convnet"
# create a png graphic of the convnet
python $CAFFE_ROOT/python/draw_net.py $CALDIR/AlexNet/train_val.prototxt $CALDIR/AlexNet/AlexNet.png

echo "done"
