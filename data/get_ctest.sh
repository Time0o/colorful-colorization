#!/bin/sh

DIR=./data

CTEST10K_DIR="$DIR/ctest10k"
CTEST10K_TXT="$CTEST10K_DIR/ctest10k.txt"

CVAL1K_DIR="$DIR/cval1k"
CVAL1K_TXT="$CVAL1K_DIR/cval1k.txt"

get_file() {
    wget -q --show-progress "$1" -O "$2"
}

copy_images() {
    echo "copying images into $1..."

    while read -r line; do
        cp "$imagenet_root/${line%% *}" "$1"
    done < "$2"
}

if [ $# -ne 1 ]; then
    echo "Usage: $(basename "$0") IMAGENET_VAL_ROOT" 2>&1
    exit 1
fi

# get image lists
mkdir -p "$CTEST10K_DIR" "$CVAL1K_DIR"

get_file \
"https://raw.githubusercontent.com/AruniRC/colorizer-fcn/master/lists/ctest10k.txt" \
"$CTEST10K_TXT"

get_file \
"https://raw.githubusercontent.com/AruniRC/colorizer-fcn/master/lists/cval1k.txt" \
"$CVAL1K_TXT"

for txt in "$CTEST10K_TXT" "$CVAL1K_TXT"; do
    sed -i 's/^\/.*\///g' "$txt"
done

# copy images
imagenet_root="$1"

copy_images "$CTEST10K_DIR" "$CTEST10K_TXT"
copy_images "$CVAL1K_DIR" "$CVAL1K_TXT"
