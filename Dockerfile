FROM --platform=linux/arm64 python:3.9

WORKDIR /app

#
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
#
COPY ./app /app/app
COPY ./src/model.py /app/src/model.py
COPY ./output/exp/weights/best.pt /app/output/exp/weights/best.pt
COPY ./data/resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
RUN pip install -r app/requiremets.txt
#--no-cache-dir --upgrade

# 
EXPOSE 80
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=80"]