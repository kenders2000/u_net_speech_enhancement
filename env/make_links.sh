#!/usr/bin/env bash
# modified from `unpack.sh` to just create the sym links, assumes data has already been extracted
# Script for unpacking clarity data downloads into the clarity root
#
# e.g. unpack.sh clarity_CEC1_data.main.v1_0.tgz <TARGET_DIR>


TARGET_DIR=/home/kenders/greenhdd/clarity_challenge/data
CLARITY_ROOT=/home/kenders/clarity_CEC1  #$(dirname $(dirname $full_path) ) # up one level from install script

# Get the top-level directory from the Clarity data
# (Should be 'clarity_CEC1_data' if unpacking main data package)
TOP_DIR=clarity_CEC1_data 

#  Unpack into the top level of the clarity directory
#mkdir -p "$TARGET_DIR"
#tar -xvzf "$PACKAGE_NAME"  -C "$TARGET_DIR" --keep-old-files

# Put a link to the data in the repo
(
    cd "$CLARITY_ROOT"/data || exit;
    rm clarity_data;
    ln -s "$TARGET_DIR"/"$TOP_DIR"/clarity_data clarity_data
)

# Add a link to the downloaded hrirs (downloaded to the repo when installed) into the data dir (not in the repo) 
(
    cd "$TARGET_DIR"/"$TOP_DIR"/clarity_data || exit;
    rm hrir;
    ln -s "$CLARITY_ROOT"/data/hrir hrir
)

# Add a file that was missing from the data package
cp "$CLARITY_ROOT"/install/missing/scenes_listeners.train.json "$CLARITY_ROOT"/data/clarity_data/metadata

