FROM registry.roqs.basf.net/base-images/python:3.8 AS build

USER root

RUN apt-get update

# install ipopt
RUN DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
  coinor-libipopt1v5 \
  coinor-libipopt-dev \
  libblas-dev \
  liblapack-dev \
  g++ \
  gfortran \
  pkg-config \
  && rm -rf /var/lib/apt/lists/*
RUN pip3 install ipopt
RUN apt-get remove -y coinor-libipopt-dev libblas-dev g++ gfortran pkg-config
