#!/bin/sh

DIR=./resources

get_file() {
    wget -q --show-progress "$1" -O "$2"
}

ZHANG_GITHUB_ROOT="https://github.com/richzhang/colorization/raw/caffe"
ZHANG_BERKELEY_ROOT="eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files"

# ab gamut
get_file \
"$ZHANG_GITHUB_ROOT/resources/pts_in_hull.npy" \
"$DIR/ab-gamut.npy"

# q prior probabilities
get_file \
"$ZHANG_GITHUB_ROOT/resources/prior_probs.npy" \
"$DIR/q-prior.npy"

# init model
get_file \
"$ZHANG_BERKELEY_ROOT/train/init_v2.caffemodel" \
"$DIR/init_v2.caffemodel"

# release models
get_file \
"$ZHANG_GITHUB_ROOT/models/colorization_deploy_v2.prototxt" \
"$DIR/colorization_deploy_v2.prototxt"

get_file \
"$ZHANG_BERKELEY_ROOT/demo_v2/colorization_release_v2.caffemodel" \
"$DIR/colorization_release_v2.caffemodel"

get_file \
"$ZHANG_BERKELEY_ROOT/demo_v2/colorization_release_v2_norebal.caffemodel" \
"$DIR/colorization_release_v2_norebal.caffemodel"
