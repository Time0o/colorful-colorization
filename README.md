# Colorful Image Colorization PyTorch

This is a from-scratch PyTorch implementation of "Colorful Image Colorization"
[1] by Zhang et al. created for the _Deep Learning in Data Science_ course at
KTH Stockholm. The distinguishing feature of this implementation is that it
makes it possible to exchange the VGG style network described in the paper with
a network based on the state of the art _DeepLabv3+_ [2] architecture.

The following sections describe in detail:
* how to install the dependencies necessary to get started with this project
* how to colorize grayscale images using pretrained network weights
* how to train the network on a new dataset
* how to run colorization and training programatically

## Prerequisites

We recommend you use Anaconda to create a virtual enviroment in which to
install the modules needed to run this program, i.e you should run:

```
conda env create --file environment.yml
```

There are some extra dependencies needed to run some of the scripts but you
should install those manually when it becomes necessary as you may not need
them.

In addition, you will need some files provided by R. Zhang, these include the
points of the ab gamut bins used to discretize the image labels and pretrained
Caffe models. Run `resources/get_resources.sh` to download these automatically.
**If you skip this step you will not be able to run the network at all, even if
you provide your own weight initialization or want to train from scratch**.

If you want to use DeepLabv3+ as a backend network you might also want to
download pretrained weights for the Xception sub-network. You can find links to
current ones
[here](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md).
Of these you should probably download
[xception_65_imagenet](http://download.tensorflow.org/models/deeplabv3_xception_2018_01_04.tar.gz).

You might also notice another shell script, `data/get_cval.sh`, that downloads
several additional resources. However, this is mainly a remnant of the
developement process and you can safely ignore it.

In order to use the pretrained weights for prediction, you will have to convert
them from Caffe/TensorFlow to PyTorch. We provide the convenience script
`scripts/convert_weights` for exactly this purpose. In order to use it you will
have to install the `caffe` Python module (if you want to convert one of the
Caffe models provided by R. Zhang) or `tensorflow` (if you want to convert a
DeepLabv3+/Xception checkpoint).

For example, in order to convert the Caffe model trained with class rebalancing
downloaded by `resources/get_resources.sh`, you can call the script like this:


```
./scripts/convert_weights vgg PYTORCH_WEIGHTS.tar \
	--weights resources/colorization_release_v2.caffemodel \
	--proto resources/colorization_deploy_v2.prototxt
```

Which will save the converted PyTorch weights to `PYTORCH_WEIGHTS.tar`.

## Colorize Images with Pretrained Weights

The easiest way to colorize several grayscale images of arbitrary size is to
place them in the same directory and colorize them in batch mode using
`scripts/convert_images`. For example, if you have placed the images in
directory `dir1` and subsequently run:

```
./scripts/convert_images predict_color \
	--input-dir dir1 \
    --output-dir dir2 \
    --model-checkpoint PYTORCH_WEIGHTS.tar \
    --gpu \
    --verbose
```

The script will colorize all images in `dir1` on the GPU and place the results
in `dir2` (with the same filenames). If your model checkpoint was not created
from a VGG style network, you will need to explicitly specify `--base-network`.
You can also choose an annealed mean temperature parameter other then the
default 0.38 with `--annealed-mean-T`. .

## References

[1] *Colorful Image Colorization*, Zhang, Richard and Isola, Phillip and Efros,
Alexei A, in ECCV 2016
([website](https://richzhang.github.io/colorization/))

[2] *Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation*, Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian
Schroff and Hartwig Adam in ECCV 2018
([arXiv](https://arxiv.org/abs/1802.02611),
[GitHub](https://github.com/tensorflow/models/tree/master/research/deeplab))
