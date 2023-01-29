import sys
import time
import yaml
import torch
import argparse
from pathlib import Path
from attrdict import AttrDict

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # projects root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from src.utils import increment_path
from src.dataset import get_data_loaders
from src.model_utils import train_one_epoch, validate_one_epoch, evaluate
from src.model import get_model_instance_segmentation

from torch.utils.tensorboard import SummaryWriter


def train(opt):
    '''
    Args:
        opt
    '''
    save_dir, epochs, noval, nosave = \
        opt.save_dir, opt.epochs, opt.noval, opt.nosave
    device = opt.device
    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Model
    model = get_model_instance_segmentation(opt.num_classes)
    # model = model.float() # mac mps
    model.to(device)  # create

    # Optimizer
    # params = [p for p in model.parameters() if p.requires_grad] # adds anything?
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    # Data Loaders
    train_loader, val_loader = get_data_loaders(opt.path_to_dataset, opt.train_coco, opt.val_coco, opt.workers, opt.batch_size)
    
    # Start training
    t0 = time.time()
    best_loss = float('inf')
    
    comment = (f'Logging results to {save_dir}\n'
               f'Starting training for {epochs} epochs...')
    layout = {
        save_dir.name: {
            'Loss_Total': ['Multiline', ['loss_total/train', 'loss_total/val']],
            'Loss_cls': ['Multiline', ['loss_classifier/train', 'loss_classifier/val']],
            'Loss_bbox': ['Multiline', ['loss_box_reg/train', 'loss_box_reg/val']],
            'Loss_mask': ['Multiline', ['loss_mask/train', 'loss_mask/val']],
            'Loss_object': ['Multiline', ['loss_objectness/train', 'loss_objectness/val']],
            'Loss_rpn_box_reg': ['Multiline', ['loss_rpn_box_reg/train', 'loss_rpn_box_reg/val']],
        },
    }
    print(comment)
    writer = SummaryWriter(save_dir, comment = comment)
    writer.add_custom_scalars(layout)
    
    for epoch in range(epochs):  # epoch ------------------------------------------------------------------
        train_one_epoch(model, optimizer, train_loader, device, epoch, epochs, writer)

        # Calculate val loss
        final_epoch = (epoch + 1 == epochs)
        if not noval or final_epoch:
            loss_dict = validate_one_epoch(model, val_loader, device, epoch, writer)
            total_loss = loss_dict['loss_total'].avg
            if total_loss < best_loss:
                best_loss = total_loss
            # Save model
            if (not nosave) or final_epoch:  # if save
                # Save last and best
                torch.save(model.state_dict(), last)
                if best_loss == total_loss:
                    torch.save(model.state_dict(), best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(model.state_dict(), w / f'epoch{epoch}.pt')
        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    print(f'\n{epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    writer.close()
    torch.cuda.empty_cache()

    # Evaluate final model on the test dataset
    coco_eval = evaluate(model, val_loader)
    # let's add hyperparameters and COCO metric
    writer.add_hparams(
        # passing hyperparameters dictionary
        {
            "best_loss": best_loss,
            "total_epochs": epochs
        },
        # passing COCO metrics
        {
            "AP/IoU/0.50-0.95/all/2000": coco_eval.stats[0],
            "AP/IoU/0.50/all/2000": coco_eval.stats[1],
            "AP/IoU/0.75/all/2000": coco_eval.stats[2],
            "AP/IoU/0.50-0.95/small/2000": coco_eval.stats[3],
            "AP/IoU/0.50-0.95/medium/2000": coco_eval.stats[4],
            "AP/IoU/0.50-0.95/large/2000": coco_eval.stats[5],
            "AR/IoU/0.50-0.95/all/100": coco_eval.stats[6],
            "AR/IoU/0.50-0.95/all/500": coco_eval.stats[7],
            "AR/IoU/0.50-0.95/all/2000": coco_eval.stats[8],
            "AR/IoU/0.50-0.95/small/2000": coco_eval.stats[9],
            "AR/IoU/0.50-0.95/medium/2000": coco_eval.stats[10],
            "AR/IoU/0.50-0.95/large/2000": coco_eval.stats[11],
        }
    )


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_config', type=str, default=None, help='path to file with train config, overwrites all params')

    parser.add_argument('--path_to_dataset', type=str, default=ROOT / 'data', help='path to dir with csv files')
    parser.add_argument('--train_coco', type=str, default=ROOT / 'data', help='path to train annotations')
    parser.add_argument('--val_coco', type=str, default=ROOT / 'data', help='path to val annotations')
    parser.add_argument('--num_classes', type=int, default=2, help='num classes to detect + background')
    parser.add_argument('--epochs', type=int, default=10, help='total training epochs')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size for training and validation')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='num dataloader workers')
    parser.add_argument('--project', default=ROOT / 'output', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--save_period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')

    return parser.parse_args()


def main(opt):
    if opt.path_to_config:
        # overwite params
        with open(opt.path_to_config) as f:
            config = yaml.safe_load(f)
            opt = AttrDict(config)
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=False)
    train(opt)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)