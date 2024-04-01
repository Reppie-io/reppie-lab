FROM python:3.11-slim

WORKDIR /reppie-labs

RUN apt-get update && apt-get -y install libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 8501