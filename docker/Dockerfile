# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-11.html
FROM everymatrix.jfrog.io/emlab-docker-remote-nvcr/nvidia/pytorch:25.11-py3

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com && \
    pip install diffusers==0.36.0 \
    tensorboard==2.20.0 \
    tqdm==4.67.1 \ 
    timm==1.0.22 \
    einops==0.8.1 \
    einx==0.3.0 \
    matplotlib==3.10.8 \
    torchdiffeq==0.2.5 \
    ema-pytorch==0.7.7 \
    accelerate==1.12.0 \
    torchinfo==1.8.0 \
    pillow==12.0.0 \
    torch-fidelity==0.3.0 \
    flax==0.12.1 \
    pandas==2.3.3

EXPOSE 6006

VOLUME [ "/workspace" ]
WORKDIR /workspace

CMD ["tensorboard", "--bind_all", "--logdir", "/workspace/.runs"]