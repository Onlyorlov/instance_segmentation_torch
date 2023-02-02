import io
import cv2
import numpy as np
import onnxruntime
from PIL import Image
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
        image, _, _ = letterbox(image, auto=False)
        print(image.shape)
        # return image_to_bts(image)

        print(image.shape)
        image = np.moveaxis(image, -1, 0)  # HWC to CHW
        print(image.shape)
        
        image = image[np.newaxis, :] # add batch dimension
        pred = self.sess.run(None, {self.input_name: image.astype(np.float32)})
        out1, out2 = pred
        print(out1.shape, out2.shape)