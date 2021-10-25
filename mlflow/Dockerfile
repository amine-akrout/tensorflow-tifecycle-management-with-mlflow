FROM python:3.8-slim-buster

RUN apt-get update -y
RUN apt-get install libyaml-cpp-dev libyaml-dev -y

RUN pip install mlflow pymysql boto3

RUN pip --no-cache-dir install --force-reinstall -I pyyaml

EXPOSE 5000