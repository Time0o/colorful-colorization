#!/bin/sh

DIR=./resources

get_file() {
    wget -q --show-progress "$1" -O "$2"
}

# ab gamut
get_file \
"https://github.com/richzhang/colorization/raw/master/resources/pts_in_hull.npy" \
"$DIR/ab-gamut.npy"

# q prior probabilities
get_file \
"https://github.com/richzhang/colorization/raw/master/resources/prior_probs.npy" \
"$DIR/q-prior.npy"

# init model
get_file \
"eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/train/init_v2.caffemodel" \
"$DIR/init_v2.caffemodel"

# release models
get_file \
"https://raw.githubusercontent.com/richzhang/colorization/master/models/colorization_deploy_v2.prototxt" \
"$DIR/colorization_deploy_v2.prototxt"

get_file \
"http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel" \
"$DIR/colorization_release_v2.caffemodel"

get_file \
"http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2_norebal.caffemodel" \
"$DIR/colorization_release_v2_norebal.caffemodel"
