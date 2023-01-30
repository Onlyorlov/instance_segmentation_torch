import io
import cv2
import torch
import numpy as np
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model_instance_segmentation(num_classes:int, pretrained:bool=True, state_dict_pth:str=None):
    # https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    if state_dict_pth:
        model.load_state_dict(torch.load(state_dict_pth, map_location=torch.device('cpu')))
        print('Loaded weights')
    return model

# from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

# def get_model_instance_segmentation(num_classes:int, pretrained:bool=True, state_dict_pth=None):
#     #load an instance segmentation model pre-trained on COCO
#     #https://github.com/fcakyon/augmented-maskrcnn/blob/master/model.py
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(
#         pretrained=pretrained,
#         pretrained_backbone=pretrained,
#         rpn_pre_nms_top_n_train=2000, # number of proposals to keep before applying NMS during training
#         rpn_pre_nms_top_n_test=2000, # number of proposals to keep before applying NMS during testing
#         rpn_post_nms_top_n_train=2000, # number of proposals to keep after applying NMS during training
#         rpn_post_nms_top_n_test=2000, # number of proposals to keep after applying NMS during testing
#         rpn_batch_size_per_image=2000, # number of anchors that are sampled during training of the RPN
#         box_detections_per_img=2000, # maximum number of detections per image, for all classes.
#         box_batch_size_per_image=2000 # number of proposals that are sampled during training of the classification head
#         )

#     # #create an anchor_generator for the FPN which by default has 5 outputs==len(anchor_sizes)
#     # anchor_generator = AnchorGenerator(
#     #     sizes=((8,), (16,), (32,), (64,), (128,)),
#     #     aspect_ratios=tuple([(0.25, 0.5, 1.0, 2.0) for _ in range(5)])
#     #     )

#     # model.rpn.anchor_generator = anchor_generator

#     # # 256 because that's the number of features that FPN returns
#     # model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
#     # print(model.rpn.score_thresh, model.rpn.nms_thresh)

#     # get the number of input features for the classifier
#     in_features = model.roi_heads.box_predictor.cls_score.in_features

#     # replace the pre-trained head with a new one
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     # now get the number of input features for the mask classifier
#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     hidden_layer = 256

#     # and replace the mask predictor with a new one
#     model.roi_heads.mask_predictor = MaskRCNNPredictor(
#         in_features_mask,
#         hidden_layer,
#         num_classes
#         )

#     if state_dict_pth:
#         model.load_state_dict(torch.load(state_dict_pth))
#         model.eval()
#     return model


class Predictor():
    def __init__(self, path_to_ckpt:str, thr:float=0.8, nms:bool=False, device:str='cpu'):
        '''
        Args:
            path_to_ckpt (str): Path to the trained model
        '''
        self.device = device
        self.model = get_model_instance_segmentation(2, pretrained=False, state_dict_pth=path_to_ckpt)
        self.model.to(self.device)
        self.model.eval()

        self.thr = thr
        self.nms = nms

    def get_predict(self, image_bytes:bytes, vis_type:str):
        if vis_type not in ("bboxes", "mask", "contour"):
            raise NotImplementedError(
                "Visualization does not support %s" % vis_type
            )

        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img = transform(image)
        with torch.no_grad():
            predict = self.model([img])[0]

        pred_t = (predict['scores']>self.thr).nonzero()
        predict['masks'] = predict['masks'][pred_t].squeeze()
        predict['boxes'] = predict['boxes'][pred_t].squeeze()
        predict['scores'] = predict['scores'][pred_t].squeeze()
        predict['labels'] = predict['labels'][pred_t].squeeze()

        if self.nms:
            predict = apply_nms(predict, iou_thresh=self.nms)
        
        if vis_type=="bboxes":
            img = self.add_bboxes(image, predict)
        elif vis_type== 'mask':
            img = self.get_mask(image, predict)
        else:
            img = self.get_contour(image, predict)
        return img

    @staticmethod
    def add_bboxes(img, prediction:dict):
        '''Adds bboxes to the PIL image
        Args:
            img (): img
            prediction (dict): model predictions
        '''
        COLORF = (212, 15, 24)
        for box in (prediction['boxes']):
            cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), COLORF, 1)
        is_success, buffer = cv2.imencode(".jpg", img)
        io_buf = io.BytesIO(buffer)
        return io_buf.getvalue()
    
    @staticmethod
    def get_mask(img, prediction:dict):
        '''Adds all masks to the PIL image
        Args:
            img (): img
            prediction (dict): model predictions
        '''
        masks = np.zeros(img.shape[:2]) # fix model inputs and throw exception!
        for mask in (prediction['masks']):
            masks+=mask.numpy().squeeze()
        # axes.imshow(masks.transpose((1, 2, 0)), aspect='auto') # [W, H, C]
        is_success, buffer = cv2.imencode(".jpg", (masks>0.5)*np.uint8(255))
        io_buf = io.BytesIO(buffer)
        return io_buf.getvalue()

    @staticmethod
    def get_contour(img, prediction:dict):
        '''Adds all contours to the PIL image
        Args:
            img (): img
            prediction (dict): model predictions
        '''
        for i in range(len(prediction['masks'])):
            # iterate over masks
            mask = prediction['masks'][i]
            mask = mask.mul(255).byte().cpu().numpy().squeeze()
            contours, _ = cv2.findContours(
                    mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, contours, -1, (255, 0, 0), 2, cv2.LINE_AA)
        is_success, buffer = cv2.imencode(".jpg", img)
        io_buf = io.BytesIO(buffer)
        return io_buf.getvalue()
    
# @staticmethod # should already be implemented in model!!!?
def apply_nms(orig_prediction:dict, iou_thresh:float=0.3):
    '''Filters original prediction
    Args:
        prediction (dict): model predictions
        iou_thresh (float): iou_thresh
    '''
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['masks'] = final_prediction['masks'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]
    
    return final_prediction