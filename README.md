# Cell instance segmentation with DL, FastAPI, Docker and Yandex cloud

## 1. Load data

```bash
python get_data.py --config configs/data_urls.yaml
```

### 1a. Create train/validation split without leakage

```bash
python new_data_split.py --config configs/data_split.yaml 
```

## 2. Train model

```bash
python train.py --config configs/train_config.yaml
```

## 3. Evaluate trained models on test set

```bash
python evaluate.py --config configs/test_config.yaml
```

## 4. Deploy

### Set model weights in `app/config.yaml` and `Dockerfile`, put backbone weights to `/data/resnet50-0676ba61.pth`. Create Docker container:

```bash
docker build --platform=linux/amd64 -t image-segmentation .
```

<!-- ROADMAP -->
## Roadmap

* [x] Fix COCO metric logging!

* [ ] COCO metric add thrs and nms

* [ ] Fix model weights location in app

* [x] Add visualization choice to an app

* [x] Add colors to visualization code

* [ ] Onnx deploy? --

* [ ] Model's params to train, test and app config!

* [ ] Epochs -> iterations

* [ ] Add config copy to exp folder

* [x] Update README

* [x] New dataset split

## Additional features

### Watch train/test losses

```bash
tensorboard --logdir=output
```

### Basic EDA in EDA.ipynb

### You can check out baseline model [here](https://bba2fr9fv1d6in16jt7b.containers.yandexcloud.net/docs)
