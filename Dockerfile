FROM python:3.7.5-slim-buster
MAINTAINER Chuan Zhang <chuan.zhang2015@gmail.com>

ENV INSTALL_PATH /CNN4Strabismus
RUN mkdir -p $INSTALL_PATH

WORKDIR $INSTALL_PATH

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

CMD gunicorn -b 0.0.0.0:8000 --access-logfile - "CNN4Strabismus.app:create_app()"
