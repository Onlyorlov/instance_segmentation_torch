# Cell instance segmentation with Yolov5, FastAPI, Docker and Yandex cloud

## 1. Load data, Create train/validation split without leakage & Transform to Yolo format

```bash
./prepare_data.sh configs/data_process.yaml
```

## 2. Train model

```bash
python yolov5/segment/train.py  --batch 4
                                --epochs 10
                                --data configs/yolo_data_config.yaml
                                --weights yolov5m-seg.pt
                                --max_det=2000
```

## 3. Evaluate trained models on test set

```bash
python yolov5/segment/val.py    --data configs/yolo_data_config.yaml
                                --weights yolov5/runs/train-seg/exp/weights/best.pt
                                --task test
                                --max_det=2000
```

## 4. Deploy

### Transform Model weights to onnx

```bash
python yolov5/export.py --weights yolov5/runs/train-seg/exp/weights/best.pt
                        --include onnx
```

### Copy best onnx model to `app/best.onnx`. Create Docker container

```bash
docker build --platform=linux/amd64 -t image-segmentation .
```

<!-- ROADMAP -->
## Roadmap

* [x] Fix model weights location in app -- copy to fix location (inside app folder)

* [x] Onnx preprocess -- letterbox

* [x] Onnx output -- thrs, nms?

* [x] Onnx postprocess --pred visualization

* [x] Fix masks: changed to lifehack solution)

* [x] Update configs

* [x] Cleanup

## Additional features

### Watch train/test losses

```bash
tensorboard --logdir=yolov5/runs
```

### Basic EDA in [EDA.ipynb](https://drive.google.com/file/d/1qZeqaf9AuR43M-k9YN8iT6JD6jkR00Tu/view?usp=sharing)

### You can check out model [here](https://bba2fr9fv1d6in16jt7b.containers.yandexcloud.net/docs)
