import os
import yaml
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from attrdict import AttrDict

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # projects root directory
from utils import TQDM_BAR_FORMAT


def make_folders(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path

def convert_coco_json_to_yolo_txt(output_path, images_dir, json_file, use_segments=True):
    make_folders(output_path)

    df_img_id = []
    df_img_name = []
    df_img_width = []
    df_img_height = []
    with open(json_file) as f:
        json_data = json.load(f)
    print('Extraction data from:',json_file)
    pbar = enumerate(json_data["images"])
    print(('\n' + '%40s %10s') % ('Image name', 'Objects'))
    pbar = tqdm(pbar, total=len(json_data["images"]), bar_format=TQDM_BAR_FORMAT)

    # write _darknet.labels, which holds names of all classes (one class per line)
    label_file = os.path.join(output_path, "_darknet.labels.txt")
    with open(label_file, "w") as f:
        for _, image in pbar:
            img_id = image["id"]
            img_name = os.path.basename(image["file_name"])
            json_images = os.path.join(images_dir,image["file_name"])
            
            shutil.copy(json_images, output_path)
            img_width = image["width"]
            img_height = image["height"]
            df_img_id.append(img_id)
            df_img_name.append(img_name)
            df_img_width.append(img_width)
            df_img_height.append(img_height)
        
            anno_in_image = [anno for anno in json_data["annotations"] if anno["image_id"] == img_id]
            anno_txt = os.path.join(output_path, img_name.split(".")[0] + ".txt")

            h, w, f = image['height'], image['width'], image['file_name']
            bboxes = []
            segments = []
            with open(anno_txt, "w") as f:
                for anno in anno_in_image:
                    category = anno["category_id"]
                    bbox_COCO = anno["bbox"]
                    #x, y, w, h = convert_bbox_coco2yolo(img_width, img_height, bbox_COCO)
                    if anno['iscrowd']:
                        continue
                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(anno['bbox'], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y
                    if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                        continue
                    #cls = coco80[anno['category_id'] - 1] if cls91to80 else anno['category_id'] - 1  # class
                    cls = anno['category_id'] - 1
                    box = [cls] + box.tolist()
                    if box not in bboxes:
                        bboxes.append(box)
                    # Segments
                    if use_segments:
                        if len(anno['segmentation']) > 1:
                            s = merge_multi_segment(anno['segmentation'])
                            s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
                        else:
                            s = [j for i in anno['segmentation'] for j in i]  # all segments concatenated
                            s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
                        s = [cls] + s
                        if s not in segments:
                            segments.append(s)

                    last_iter=len(bboxes)-1
                    line = *(segments[last_iter] if use_segments else bboxes[last_iter]),  # cls, box or segments
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
                pbar.set_description(('%40s %10s') % (img_name, len(bboxes)))
    
    print("Creating category_id and category name in darknet.labels")
    with open(label_file, "w") as f:
        pbar = enumerate(json_data["categories"])
        print(('%20s') % ('Category'))
        pbar = tqdm(pbar, total=len(json_data["categories"]), bar_format=TQDM_BAR_FORMAT)
        for _, category in pbar:
            category_name = category["name"]
            f.write(f"{category_name}\n")
            pbar.set_description(('%20s') % (category_name))

    print("Finish")


def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance. 
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all 
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...], 
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0]:idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=ROOT / 'configs/data_preprocess.yaml', help='path to config file')

    return parser.parse_args()

def main(opt):
    print(f'Loading data')
    with open(opt.config) as f:
        config = yaml.safe_load(f)
        config = AttrDict(config)
    convert_coco_json_to_yolo_txt(config.output_train, config.train_val_img, config.new_paths.train)
    convert_coco_json_to_yolo_txt(config.output_val, config.train_val_img, config.new_paths.val)
    if config.test_json:
        convert_coco_json_to_yolo_txt(config.output_test, config.test_img, config.test_json)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)