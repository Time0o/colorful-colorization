#!/bin/bash

DIR=./data

CVAL48K_DIR="$DIR/cval48k"
CVAL10K_DIR="$DIR/cval10k"
CVAL1K_DIR="$DIR/cval1k"

LABELFILE="labels.txt"

get_file() {
    wget -q --show-progress "$1" -O "$2"
}

CTEST_GITHUB_ROOT="https://github.com/AruniRC/colorizer-fcn/raw/master"

copy_images() {
    echo "copying images into $1..."

    while read -r line; do
        cp "$imagenet_val_root/${line%% *}" "$1"
    done < "$1/$LABELFILE"
}

if [ $# -ne 2 ]; then
    echo "Usage: $(basename "$0") IMAGENET_VAL_ROOT IMAGENET_VAL_LABELS" 2>&1
    exit 1
fi

imagenet_val_root="$1"
imagenet_val_labels="$2"

# create image directories
mkdir -p "$CVAL48K_DIR" "$CVAL10K_DIR" "$CVAL1K_DIR"

# get image lists
cat "$imagenet_val_labels" | head -n 48000 > "$CVAL48K_DIR/$LABELFILE"

get_file \
"$CTEST_GITHUB_ROOT/lists/ctest10k.txt" \
"$CVAL10K_DIR/$LABELFILE"

get_file \
"$CTEST_GITHUB_ROOT/lists/cval1k.txt" \
"$CVAL1K_DIR/$LABELFILE"

for dir in "$CVAL48K_DIR" "$CVAL10K_DIR" "$CVAL1K_DIR"; do
    labelfile="$dir/$LABELFILE"
    sed -i 's/^\/.*\///g' "$labelfile"
    sort "$labelfile" -o "$labelfile"
done

# copy images
copy_images "$CVAL48K_DIR"
copy_images "$CVAL10K_DIR"
copy_images "$CVAL1K_DIR"
