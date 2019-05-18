# Colorful Image Colorization PyTorch

This is a from-scratch PyTorch implementation of "Colorful Image Colorization"
[1] by Zhang et al. created for the _Deep Learning in Data Science_ course at
KTH Stockholm.

The following sections describe in detail:
* how to install the dependencies necessary to get started with this project
* how to colorize grayscale images using pretrained network weights
* how to train the network on a new dataset
* how to run colorization programatically

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

You might also notice another shell script, `data/get_cval.sh`, that downloads
several additional resources. However, this is mainly a remnant of the
developement process and you can safely ignore it.

In order to use the pretrained weights for prediction, you will have to convert
them from Caffe to PyTorch. We provide the convenience script
`scripts/convert_weights` for exactly this purpose. In order to use it you will
have to install the `caffe` Python module (if you want to convert one of the
Caffe models provided by R. Zhang)

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
in `dir2` (with the same filenames). You can choose an annealed mean
temperature parameter other then the default 0.38 with `--annealed-mean-T`. .

## Train the Network

### Prepare a Dataset

If you intend to train the network on your own dataset, you might want to use
the convenience scripts `scripts/prepare_dataset` to convert it into a form
suitable for training. For example, if all your images are stored in a
directory tree similar to this one:

```
dir1/
├── subdir1
│   ├── img1.JPEG
│   ├── img2.JPEG
│   └── ...
├── subdir2
│   ├── img1.JPEG
│   ├── img2.JPEG
│   └── ...
└── ...

```

you may want to run:

```
./scripts/prepare_dataset dir1 \
    --flatten \
    --purge \
    --clean \
    --file-ext JPEG \
    --val-split 0.2 \
    --test-split 0.1 \
    --resize-height 256 \
    --resize-width 256 \
    --verbose
```

The script will first recursively look for images files with the extension
`.JPEG` in `dir1` and remove all other files and those images that cannot be
read or converted to RGB. It will then resize all remaining images to 256x256
and randomly place them in the newly created subdirectories `train`, `val` and
`test` using a 70/20/10 split.

Note that this will take a while for large datasets since every single image
has to be read into memory. If your images already have the desired size (this
does not necessarily have to be 256x256, the network is fully convolutional and
can train on images of arbitrary size) and you are sure that none of them are
corrupted, you don't have to use the `--resize-height/--resize-width` and
`--clean` arguments which will speed up the process considerably.

### Run the Training

To train the network on your dataset you can use the script
`scripts/run_training`. The script accepts command line arguments that control
e.g. the duration of the training and where/how often logfiles and model
checkpoints are written. More specific settings like dataloader configuration,
network type and optimizer settings need to be specified via a configuration
file which is essentially a nested directory of Python objects converted to
JSON. Most likely you will want to use `config/default.json` and provide
specific settings or override some defaults in a separate JSON file. See
`config/vgg.json` for an example.

Once you have decided on a configuration file you can run the script as follows:

```
./scripts/run_training \
    --config YOUR_CONFIG.json \
    --default-config config/default.json \
    --data-dir dir1 \
    --checkpoint-dir YOUR_CHECKPOINT_DIR \
    --log-file YOUR_LOG_FILE.txt \
    --iterations ITERATIONS \
    --iterations-till-checkpoint ITERATIONS_TILL_CHECKPOINT \
    --init-model-checkpoint INIT_MODEL-CHECKPOINT.tar
```

This will recursively merge the configurations in `YOUR_CONFIG.json` and
`config/default.json` and then train on the the images in `dir1` for
`ITERATIONS` iterations (batches). Every `ITERATIONS_TILL_CHECKPOINT`
iterations, an intermediate model checkpoint will be written to
`YOUR_CHECKPOINT_DIR`. Specifying `--init-model-checkpoint` is optional but
useful if you want to finetune the network from some pretrained set of weights.

You can also continue training from an arbitrary training checkpoint using the
`--continue-training` flag which will load network weights and optimizer state
from `INIT_MODEL_CHECKPOINT.tar` (which has to be a checkpoint created by a
previous run of `scripts/run_training`) and pick the training up from the last
training iteration (thus `ITERATIONS` still specifies the total number of
training iterations).

## Colorize Images Programmatically

Colorizing images programmatically using our implementation is very simple. You
first need to instantiate the network itself:

```python
from colorization.modules.colorization_network import ColorizationNetwork

network = ColorizationNetwork(annealed_mean_T=0.38, device='gpu')
```

The parameters should be self explanatory (and are in this case optional), use
`device='cpu'` if you plan to run the network on the GPU.

You will then need to wrap the network in an instance of `ColorizationModel`
which implements (among other things) checkpoint saving/loading:

```python
from colorization.colorization_model import ColorizationModel

model = ColorizationModel(network)
model.load_checkpoint('YOUR_CHECKPOINT_DIR/checkpoint_final.tar')
```

In order to colorize a grayscale image you should then:
* load it into a numpy array
* resize a copy of it to 224x224 (this is not strictly necessary but produces
  better results)
* convert it to a torch tensor
* pass it through the model
* reassemble the result

All of this is already implemented in a convenience function:

```python
from colorization.util.image import imread, predict_color

img = imread('YOUR_IMAGE.jpg')
img_colorized = predict_color(img)
```

## References

[1] *Colorful Image Colorization*, Zhang, Richard and Isola, Phillip and Efros,
Alexei A, in ECCV 2016
([website](https://richzhang.github.io/colorization/))
