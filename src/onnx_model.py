import io
import numpy as np
import onnxruntime
from PIL import Image

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
        image = Image.open(io.BytesIO(image_bytes))
        assert image.size != (520, 704), "Input format not supported"
        image = np.float32(image)
        print(image.shape)
        image = image[:, :, np.newaxis] # add channel dim for an image
        print(image.shape)
        image = np.moveaxis(image, -1, 0)  # HWC to CHW
        print(image.shape)
        
        image = image[np.newaxis, :] # add batch dimension
        pred = self.sess.run(None, {self.input_name: image})
        boxes, labels, scores, masks = pred
        print(pred)
        print(type(boxes), scores)

import numpy as np

def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep