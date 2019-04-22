# Colorful Image Colorization PyTorch Implementation

This is a from scratch PyTorch implementation of "Colorful Image Colorization"
by Zhang et al.

## Prerequisites

Install python dependencies by running `pip -r requirements.txt`.

You will need to install PyTorch separately, see the PyTorch homepage for
instructions.

Note that in order to load pretrained model parameters from the Caffe model
provided by Zhang you will also have to install Caffe (see for example [these
instructions]( http://caffe.berkeleyvision.org/install_apt_debian.html) if
you're running Debian).

In addition, you will need some files provided by the authors of the paper,
these include the points of the ab gamut bins used to discretize the image
labels and their pretrained Caffe models. Run `resources/get_resources.sh` to
download these automatically.

## Training The Model

To train the model from scratch (this runs but is currently too slow and might
still be buggy) run `python run_training --config config/default.json`.

## Running Predictions

You can run predictions using either the model you trained yourself or a model
initialized from a pretrained Caffe model. For the latter you can run:

```
./run_evaluation.py --pretrain-proto resources/colorization_deploy_v2.prototxt \
                    --pretrain-model resources/colorization_release_v2.caffemodel \
                    --input-image SOME_IMAGE
                    --output-image SOME_PATH
```

To predict color channels for a single image, replacing `SOME_IMAGE` and
`SOME_PATH` by actual file paths (if the provided input image is a color image
it will be converted to grayscale automatically)
