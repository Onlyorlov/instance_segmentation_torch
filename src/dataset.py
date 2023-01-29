import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader

from src.utils import fig2img


class myCOCODataset(Dataset):
    def __init__(self, root:str, annotation:str, transforms=None):
        """
        Args:
            root (string): Path to folder with images
            annotation (string): Path to json file with labels
            transforms (optional): Torch.Transforms?
        """
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index:int):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Adding segmentation masks.
        masks = []
        for i in range(num_objs):
            masks.append(coco.annToMask(coco_annotation[i]))
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["masks"] = masks
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)
        return img, my_annotation

    def __len__(self):
        return len(self.ids)
    
    def visualize(self, index:int, return_mask:bool=False):
        # https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]

        figure = plt.figure(frameon=False)
        axes = plt.Axes(figure, [0., 0., 1., 1.])
        axes.set_axis_off()
        figure.add_axes(axes)
        if not return_mask:
            image = np.array(Image.open(os.path.join(self.root, path)))
            axes.imshow(image, interpolation='nearest', aspect='auto')
            coco.showAnns(coco_annotation, draw_bbox=True)
            plt.close()
            return fig2img(figure)
        else:
            mask = coco.annToMask(coco_annotation[0])
            for i in range(len(coco_annotation)):
                mask += coco.annToMask(coco_annotation[i])
            axes.imshow(mask, aspect='auto')
            plt.close()
            return fig2img(figure)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_data_loaders(
    path_to_dataset_imgs:str,
    path_to_train_json:str,
    path_to_val_json:str,
    num_workers:int = 0,
    batch_size:int =1,
    train_augs:bool=False
    ):
    '''
    Args:
        path_to_dataset_imgs (str): Path to dir with images
        path_to_train_json (str): Path to json with train annotations
        path_to_val_json (str): Path to json with val annotations
        num_workers (int): number of workers for dataloaders (check available CPU cores)
        batch_size (int): batch size
        train_augs (bool): Use augmentations when training
    '''
    dataset_train = myCOCODataset(
        root=path_to_dataset_imgs,
        annotation=path_to_train_json,
        transforms=get_transform(train=train_augs)
        )
    dataset_valid = myCOCODataset(
        root=path_to_dataset_imgs,
        annotation=path_to_val_json,
        transforms=get_transform(train=False)
        )

    loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=collate_fn
        )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn
        )
    return loader_train, loader_valid

def get_data_loader(path_to_dataset_imgs:str, path_to_json:str, num_workers:int = 0, batch_size:int =1):
    '''
    Args:
        path_to_dataset_imgs (str): Path to dir with images
        path_to_json (str): Path to json with annotations
        num_workers (int): number of workers for dataloaders (check available CPU cores)
        batch_size (int): batch size
    '''
    dataset = myCOCODataset(
        root=path_to_dataset_imgs,
        annotation=path_to_json,
        transforms=get_transform(train=False)
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn
        )
    return loader


# These are slight redefinitions of torch.transformation classes
# The difference is that they handle the target and the mask
# Copied from Abishek, added new ones
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class VerticalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, _ = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-2)
        return image, target

class HorizontalFlip:
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            _, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            target["masks"] = target["masks"].flip(-1)
        return image, target

# class Normalize: # should already be inside mask-rcnn 
#     def __call__(self, image, target):
#         image = F.normalize(image, config.RESNET_MEAN, config.RESNET_STD)
#         return image, target

class ToTensor:
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target

def get_transform(train=False):
    transforms = [ToTensor()]
    
    # Data augmentation for train
    if train: 
        transforms.append(HorizontalFlip(0.5))
        transforms.append(VerticalFlip(0.5))
    return Compose(transforms)