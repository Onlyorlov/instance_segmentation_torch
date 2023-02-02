# Cell instance segmentation with DL, FastAPI, Docker and Yandex cloud

## 1. Load data, Create train/validation split without leakage & Transform to Yolo format

```bash
./preprocess_sript.sh --config=configs/data_preprocess.yaml
```

## 2. Train model

```bash
python yolov5/segment/train.py  --batch 4
                                --epochs 10
                                --data configs/yolo_data_config.yaml
                                --weights yolov5m-seg.pt
                                --name custom-dataset
```

## 3. Evaluate trained models on test set

```bash
python yolov5/segment/val.py    --data configs/yolo_data_config.yaml
                                --weights runs/best_copy.pt
                                --task test                        
```

## 4. Deploy

### Transform Model weights to onnx

```bash
python yolov5/export.py --weights /content/yolov5/runs/train-seg/custom-dataset/weights/best.pt
                        --include onnx
```

### Copy best onnx model to `app/best.onnx`. Create Docker container

```bash
docker build --platform=linux/amd64 -t image-segmentation .
```

<!-- ROADMAP -->
## Roadmap

* [ ] COCO metric add thrs and nms

* [x] Fix model weights location in app -- copy to fix location (inside app folder)

* [x] Onnx preprocess -- letterbox

* [ ] Onnx output -- thrs, nms?

* [ ] Onnx postprocess --pred visualization

* [ ] Update configs

## Additional features

### Watch train/test losses

```bash
tensorboard --logdir=runs
```

### Basic EDA in EDA.ipynb

<!-- ### You can check out baseline model [here](https://bba2fr9fv1d6in16jt7b.containers.yandexcloud.net/docs) -->
