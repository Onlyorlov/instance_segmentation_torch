import yaml
import json
import argparse
import numpy as np
import pandas as pd

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # projects root directory


def make_new_split(old_paths:dict, new_paths:dict, seed:int):
    '''
    Args:
        old_paths (dict): dict with existing split's jsons paths
        old_paths (dict): dict to new split's jsons path
        seed (int): to fix data split
    '''
    np.random.seed(seed)

    df_old_train = get_df(old_paths['train'])
    df_old_val = get_df(old_paths['val'])
    df = pd.concat([df_old_train, df_old_val])

    val_choices = {}
    for cell_type in df.cell_type.unique():
        val_choices[cell_type] = np.random.choice(df[df.cell_type == cell_type].well.unique(), 1).item()
    print('Val choice: ', val_choices)
    dfs_images = {}
    dfs_annotations = {}

    dfs_val = []
    for key, value in val_choices.items():
        dfs_val.append(df[(df.cell_type == key) & (df.well == value)])
    dfs_images['val'] = pd.concat(dfs_val)
    val_ids = dfs_images['val'].id.values
    dfs_images['train'] = df[~df.id.isin(val_ids)]

    df_anno = pd.concat([get_df(old_paths['train'], 'annotations'), get_df(old_paths['val'], 'annotations')])
    dfs_annotations['val'] = df_anno[df_anno.image_id.isin(val_ids)]
    dfs_annotations['train'] = df_anno[~df_anno.image_id.isin(val_ids)]

    # write modified files
    for task in ['train', 'val']:
        with open(old_paths[task], 'r') as f:
            data = json.load(f)
        data['images'] = dfs_images[task][dfs_images[task].columns.difference(['cell_type', 'well'])].to_dict('records')
        data['annotations'] = dfs_annotations[task].to_dict('records')
        with open(new_paths[task], 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    print(f"Prev img distr: train-{df_old_train.shape[0]}, val-{df_old_val.shape[0]}")
    print(f"New img distr: train-{dfs_images['train'].shape[0]}, val-{dfs_images['val'].shape[0]}")

def get_df(json_path:str, data_type:str='images'):
    with open(json_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data[data_type])
    if data_type == 'images':
        df['cell_type'] = df.file_name.apply(lambda x: x.split('_')[0])
        df['well'] = df.file_name.apply(lambda x: x.split('_')[2])
    return df
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=ROOT / 'data_urls.yaml', help='path to config file with files locations')
    return parser.parse_args()

def main(opt):
    print(f'Loading data')
    with open(opt.config) as f:
        config = yaml.safe_load(f)
    make_new_split(config['old_paths'], config['new_paths'], config['seed'])
    print(f"Split jsons saved to {config['new_paths']}")

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)