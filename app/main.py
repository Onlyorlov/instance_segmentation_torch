#main.py

import yaml
from attrdict import AttrDict
from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware

from src.model import Predictor


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

model = Predictor(settings.path_to_model, thr=0.6, nms=0.3, device ='cpu')

@app.get("/")
def home():
    return {"health_check": "OK", "model_version": settings.app_version}

@app.post("/predict")
def predict(file: UploadFile):
    # save to images tmp
    image_bytes = file.file.read()
    image_bytes = model.get_predict(image_bytes, 'contour') #mask, contour, bboxes
    return Response(content=image_bytes, media_type="image/png")

