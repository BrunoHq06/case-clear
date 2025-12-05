FROM python:3.11-bullseye

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt