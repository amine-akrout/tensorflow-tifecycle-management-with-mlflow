FROM python:3.8
COPY . /

WORKDIR /

RUN apt-get update -y
RUN apt-get install libyaml-cpp-dev libyaml-dev -y

RUN pip install mlflow pymysql

RUN pip --no-cache-dir install --force-reinstall -I pyyaml

ENTRYPOINT ["mlflow", "server"]