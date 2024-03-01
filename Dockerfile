FROM python:3.11-slim

# RUN mkrdir /app

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

EXPOSE 8501