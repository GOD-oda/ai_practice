FROM python:3.9-alpine

WORKDIR /var/www/html

RUN apt-get update & apt-get -y install vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools
