FROM python:3.10.6-buster

WORKDIR /app
COPY shark_id shark_id
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY raw_data/model raw_data/model

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install .

CMD uvicorn shark_id.api.fast:app --host 0.0.0.0 --port $PORT
