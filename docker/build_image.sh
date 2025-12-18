#!/bin/bash
IMAGE_URL=everymatrix.jfrog.io/emlab-docker/emai/tim:train-20251213
docker build --progress=plain -t=$IMAGE_URL .