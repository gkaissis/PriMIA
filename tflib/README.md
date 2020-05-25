# tflib
All TF-dependent code should live here.


# Docker
For consistent development environments / physical GPU use with TF.

## Prerequsities

- nvidia-docker

## Build an image.

`docker build -t tflib:latest`

## Run the image with mounted tflib

Run default command

`docker run --gpus all -it -v ${PWD}:/opt/4P tflib:latest`

Override default command

`docker run --gpus all -it -v ${PWD}:/opt/4P tflib:latest "python tflib/main.py"`

Debug interactively

`docker run --gpus all -it -v ${PWD}:/opt/4P tflib:latest "/bin/bash"`

## TensorBoard

Add `-v /local/path:/opt/tensorboard` when running and configure script to output TensorBoard logs to `/opt/tensorboard`.
