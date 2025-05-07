FROM docker.io/nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
  software-properties-common \
  python3 \
  python3-pip \
  python3-dev \
  build-essential \
  gdal-bin \
  libgdal-dev

RUN pip3 install --upgrade pip
RUN pip3 install numpy pillow
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install gdal[numpy]=="$(gdal-config --version)"

# current default weights (torchvision.models.ResNet50_Weights.DEFAULT aka ResNet50_Weights.IMAGENET1K_V2)
ADD https://download.pytorch.org/models/resnet50-11ad3fa6.pth /root/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth

WORKDIR /src