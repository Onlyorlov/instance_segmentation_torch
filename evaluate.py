import yaml
import argparse
from attrdict import AttrDict

from src.dataset import get_data_loader
from src.model_utils import evaluate
from src.model import get_model_instance_segmentation

def eval(opt):
    '''
    Args:
        opt
    '''
    # Model
    model = get_model_instance_segmentation(opt.num_classes, state_dict_pth=opt.model_pth)
    
    # Data Loader
    loader = get_data_loader(opt.path_to_dataset, opt.coco_json, opt.workers, opt.batch_size)

    # Evaluate model on the dataset
    coco_eval_res = evaluate(model, loader, opt.thr, opt.nms)
    print(coco_eval_res)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='path to file with eval config')

    return parser.parse_args()


def main(opt):
    with open(opt.config) as f:
        config = yaml.safe_load(f)
        config = AttrDict(config)
    eval(config)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)