import sys
import time
import math
import torch
import torchvision
from tqdm import tqdm

from src.coco_eval import CocoEvaluator
from src.utils import AvgMeter, TQDM_BAR_FORMAT
from src.coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, epochs, writer):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        # The learning rate is increased linearly over the warm-up period(first epoch).
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    pbar = enumerate(data_loader)
    loss_names = ['loss_total', 'loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']
    print(('\n' + '%11s' * 2 + '%18s' * 6) % ('Epoch', 'GPU_mem', *loss_names))
    pbar = tqdm(pbar, total=len(data_loader), bar_format=TQDM_BAR_FORMAT)  # progress bar
    optimizer.zero_grad()
    meter_loss_dict = {name:AvgMeter() for name in loss_names}
    for _, (images, targets) in pbar:  # batch -------------------------------------------------------------
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        # Update mean losses
        meter_loss_dict['loss_total'].update(loss_value)
        for key, value in loss_dict.items():
            meter_loss_dict[key].update(value.item())

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(losses)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # Display
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
        pbar.set_description(('%11s' * 2 + '%18.4g' * 6) %
                                (f'{epoch}/{epochs - 1}', mem, *[l.avg for l in meter_loss_dict.values()]))
        # end batch ------------------------------------------------------------------------------------------------
    # Log
    for key, value in meter_loss_dict.items():
        writer.add_scalar(f'{key}/train', value.avg, epoch)
    writer.add_scalar('lr', optimizer.param_groups[0]["lr"], epoch)
    return meter_loss_dict

def validate_one_epoch(model, data_loader, device, epoch, writer):
    pbar = enumerate(data_loader)
    loss_names = ['loss_total', 'loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg']
    print(('%11s' + '%18s' * 6) % ('Validation:', *loss_names))
    pbar = tqdm(pbar, total=len(data_loader), bar_format=TQDM_BAR_FORMAT)  # progress bar

    meter_loss_dict = {name:AvgMeter() for name in loss_names}
    with torch.no_grad():
        # model.eval() changes model behavior and it does not return losses
        # https://stackoverflow.com/questions/71288513/how-can-i-determine-validation-loss-for-faster-rcnn-pytorch
        for _, (images, targets) in pbar:  # batch -------------------------------------------------------------
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            # Update mean losses
            meter_loss_dict['loss_total'].update(loss_value)
            for key, value in loss_dict.items():
                meter_loss_dict[key].update(value.item())

            # Display
            pbar.set_description(('%11s' * 1 + '%18.4g' * 6) %
                                    ('', *[l.avg for l in meter_loss_dict.values()]))
            # end batch ------------------------------------------------------------------------------------------------
    # Log
    for key, value in meter_loss_dict.items():
        writer.add_scalar(f'{key}/val', value.avg, epoch)
    return meter_loss_dict


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader): # add thrs? nms?
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    model.to(cpu_device)
    print('\n')
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    # https://stackoverflow.com/questions/52839368/understanding-coco-evaluation-maximum-detections
    coco_evaluator = CocoEvaluator(coco, iou_types)

    print(('%30s' + '%15s' * 2) % ('Calculating COCO metrics: ', 'model_predict', 'evaluation'))
    pbar = enumerate(data_loader)
    pbar = tqdm(pbar, total=len(data_loader), bar_format=TQDM_BAR_FORMAT)  # progress bar
    predict_time, eval_time = AvgMeter(), AvgMeter()
    for _, (images, targets) in pbar:
        images = list(img.to(cpu_device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        with torch.no_grad():
            outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        predict_time.update(time.time() - model_time)

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        eval_time.update(time.time() - evaluator_time)
        # Display
        pbar.set_description(('%30s' + '%15.4g' * 2) %
                                ('', predict_time.avg, eval_time.avg))


    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator