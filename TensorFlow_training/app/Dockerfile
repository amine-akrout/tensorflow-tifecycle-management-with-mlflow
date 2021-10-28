FROM python:3.8-slim-buster
RUN apt-get update -y
RUN apt-get install libyaml-cpp-dev libyaml-dev -y

WORKDIR /app

RUN pip install streamlit requests
RUN pip --no-cache-dir install --force-reinstall -I pyyaml


EXPOSE 9000