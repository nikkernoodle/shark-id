FROM python:3.10.6-buster

COPY shark_id /shark_id
COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
COPY raw_data/model /raw_data/model

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install .

#to build it locally
# CMD uvicorn shark_id.api.fast:app --host 0.0.0.0 --port 8000

#to build it for the cloud
CMD uvicorn shark_id.api.fast:app --host 0.0.0.0 --port $PORT
