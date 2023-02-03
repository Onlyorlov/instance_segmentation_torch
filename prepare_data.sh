#!/bin/bash
# A Simple Shell Script To Combine all preprocess steps and create needed files in default folders
# Usage: ./preprocess_script.sh --path_to_zip_archive=PATH_TO_ARCHIVE

python src/get_data.py --config "$@" && python src/new_data_split.py --config "$@" && python src/coco_to_yolo.py --config "$@"