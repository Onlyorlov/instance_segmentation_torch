import os
import yaml
import zipfile
import argparse
from pathlib import Path
from attrdict import AttrDict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # projects root directory

def create_dataset(path_to_dataset:str, config_pth:str):
    '''
    Args:
        path_to_dataset (string): Path to desired dataset location
        config_pth (string): Path to dataset urls
    '''
    with open(config_pth) as f:
        config = yaml.safe_load(f)
        config = AttrDict(config)
    
    if not os.path.exists(path_to_dataset):
        os.makedirs(path_to_dataset)
    os.chdir(path_to_dataset)

    fnm = 'images.zip'
    os.system(f'wget {config.images} -O {fnm}')
    with zipfile.ZipFile(fnm, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(fnm)
    
    for split_url in config.annotations:
        os.system(f'wget {split_url}')
    
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', type=str, default=ROOT / 'data', help='dir where to put dataset')
    parser.add_argument('--config', type=str, default=ROOT / 'data_urls.yaml', help='path to config file with data urls')

    return parser.parse_args()

def main(opt):
    print(f'Loading data')
    create_dataset(opt.path_to_dataset, opt.config)
    print(f'Data saved to {opt.path_to_dataset}')

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)