FROM --platform=linux/amd64 python:3.9

#
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

#
WORKDIR /app
COPY ./app/requiremets.txt /app/app/requiremets.txt
RUN pip install -r app/requiremets.txt

COPY ./app /app/app
COPY ./app/start.sh /app/start.sh
RUN chmod +x ./start.sh
COPY ./src /app/src

EXPOSE 8080 80

#--no-cache-dir --upgrade
CMD ["./start.sh"]