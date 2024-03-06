FROM python:3.11-slim

WORKDIR /reppie-labs

COPY . .

RUN pip3 install -r requirements.txt