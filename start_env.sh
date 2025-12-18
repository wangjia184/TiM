#!/bin/bash
 
docker rm -f jerry_train
docker run -d --name jerry_train \
    -p 6006:6006 \
    -p 2222:22 \
    --gpus all \
    --runtime=nvidia \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v $(pwd):/workspace \
    -v $HOME/.ssh/authorized_keys:/authorized_keys:ro \
    -e GIT_EMAIL=jerry.wang@everymatrix.com \
    -e GIT_NAME=jerry.wang \
    -e LD_LIBRARY_PATH=/workspace \
    everymatrix.jfrog.io/emlab-docker/emai/tim:train-20251213
 
 docker exec -ti jerry_train /bin/bash