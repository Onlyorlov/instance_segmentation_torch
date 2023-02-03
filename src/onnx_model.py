import numpy as np
import onnxruntime
from src.utils import (
    letterbox, bts_to_img, image_to_bts,
    nms, Annotator, colors, process_mask,
    scale_boxes, xywh2xyxy
    )

class Predictor():
    def __init__(self, onnx_pth:str, conf_thres:float=0.8, iou_thres:bool=False):
        self.sess = onnxruntime.InferenceSession(onnx_pth)
        self.input_name = self.sess.get_inputs()[0].name

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def get_predict(self, image_bytes:bytes, vis_type:str, hide_labels:bool = False):
        if vis_type not in ("bboxes", "mask", "mask_bboxes"):
            raise NotImplementedError(
                "Visualization does not support %s" % vis_type
            )
        
        image = bts_to_img(image_bytes)
        img, _, _ = letterbox(image, auto=False)
        img = img.astype(np.float32)
        img /= 255
        img = np.moveaxis(img, -1, 0)  # HWC to CHW
        img = img[np.newaxis, :] # add batch dimension
        pred, proto = self.sess.run(None, {self.input_name: img})
        pred, proto = pred[0], proto[0] #?
        pred = xywh2xyxy(pred)

        if self.conf_thres: # THRS:
            keep = pred[:, 4] > self.conf_thres
            pred = pred[keep]
        if self.iou_thres: # NMS
            keep = nms(pred, self.iou_thres)
            pred = pred[keep]

        # Process predictions
        if vis_type == "mask_bboxes":
            hide_labels=True
        annotator = Annotator(image)
        if len(pred):
            if vis_type != "bboxes":
                masks = process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
                annotator.masks(
                    masks,
                    colors=[colors(i, True) for i in range(len(pred))],
                    im_gpu=img[0])
            
            if vis_type != "mask":
                pred[:, :4] = scale_boxes(img.shape[2:], pred[:, :4], image.shape).round()  # rescale boxes to im0 size
                # Write results
                for _, (*xyxy, conf, cls) in enumerate(reversed(pred[:, :6])):
                    label = None if hide_labels else 'Cell'
                    annotator.box_label(xyxy, label, color=(255,255,255), txt_color=(0,0,0))

        return image_to_bts(annotator.result()*255)
