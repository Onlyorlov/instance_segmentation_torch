import io
import cv2
import numpy as np
import onnxruntime
from src.utils import letterbox, bts_to_img, image_to_bts

class Predictor():
    def __init__(self, onnx_pth:str, thr:float=0.8, nms:bool=False, device=None):
        self.sess = onnxruntime.InferenceSession(onnx_pth)
        self.input_name = self.sess.get_inputs()[0].name

        self.thr = thr
        self.nms = nms

    def get_predict(self, image_bytes:bytes, vis_type:str):
        if vis_type not in ("bboxes", "mask", "contour"):
            raise NotImplementedError(
                "Visualization does not support %s" % vis_type
            )
        image = bts_to_img(image_bytes)
        img, _, _ = letterbox(image, auto=False)
        img /= 255
        # return image_to_bts(image)
        img = np.moveaxis(img, -1, 0)  # HWC to CHW
        print(img.shape)
        img = img[np.newaxis, :] # add batch dimension
        pred, proto = self.sess.run(None, {self.input_name: img.astype(np.float32)})

        # THRS!: cls*conf
        # NMS
        if nms: # write my own
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Process predictions
        det = pred
        im0 = image
        im = img
        hide_labels = False

        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        if len(det):
            masks = process_mask(proto, det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            # Mask plotting
            annotator.masks(
                masks,
                colors=[colors(x, True) for x in det[:, 5]],
                im_gpu=im)

            # Write results
            for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                c = int(cls)  # integer class
                label = None if hide_labels else 'Cell'
                annotator.box_label(xyxy, label, color=colors(c, True))
                # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)

        return image_to_bts(im0)
