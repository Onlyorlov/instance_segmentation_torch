# FROM --platform=linux/arm64 python:3.9
FROM --platform=linux/amd64 python:3.9

WORKDIR /app

#
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
#
COPY ./app /app/app
COPY ./app/start.sh /app/start.sh
RUN chmod +x ./start.sh
COPY ./src/model.py /app/src/model.py
COPY ./output/exp/weights/best.pt /app/output/exp/weights/best.pt
COPY ./data/resnet50-0676ba61.pth /root/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth
RUN pip install -r app/requiremets.txt
#--no-cache-dir --upgrade
CMD ["./start.sh"]