#main.py

import yaml
from attrdict import AttrDict
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, Response, Query

from src.onnx_model import Predictor


with open('app/config.yaml') as f:
    config = yaml.safe_load(f)
    settings = AttrDict(config)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Predictor(
    settings.path_to_model, conf_thres=settings.conf_thres,
    iou_thres=settings.conf_thres, limit=settings.vis_limit)

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": settings.app_version}

@app.post("/predict")
def predict(file: UploadFile,
            vis_type: str = Query("bboxes", enum=["bboxes", "mask", "mask_bboxes"])):
    # save to images tmp
    image_bytes = file.file.read()
    image_bytes = model.get_predict(image_bytes, vis_type) # mask, contour, bboxes
    return Response(content=image_bytes, media_type="image/png")
