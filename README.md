# Cell instance segmentation with DL, FastAPI, Docker and Heroku

## 1. Load data

```bash
python get_data.py --config configs/data_urls.yaml
```

## 2. Train model

```bash
python train.py --config configs/train_config.yaml
```

## 3. Evaluate trained models on test set

```bash
python evaluate.py --config configs/test_config.yaml
```

## 4. Set model weights in `app/config.yaml`, put backbone weights to `/data/resnet50-0676ba61.pth`

## 5. Create Docker container

### For mac

```bash
docker build --platform=linux/arm64 -t image-segmentation .
```

### For deployment

```bash
docker build --platform=linux/amd64 -t image-segmentation .
```

## 6. Run docker-compose

```bash
docker-compose up
```

<!-- ROADMAP -->
## Roadmap

* [ ] Fix COCO metric logging!

* [ ] Fix model weights location in app

* [ ] Model's params to train, test and app config!

* [ ] Add config copy to exp folder

* [ ] Update README

## Additional features

### Watch train/test losses

```bash
tensorboard --logdir=output
```

### Basic EDA in EDA.ipynb

### You can check out baseline model [here](https://instance-segmentation.herokuapp.com/docs)
