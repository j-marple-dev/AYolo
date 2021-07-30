FROM nvcr.io/nvidia/tensorrt:20.03-py3


WORKDIR /usr/src/yolo
COPY . . 
RUN apt-get update && apt-get install -y libgl1-mesa-dev && apt-get -y install jq
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt 
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100 
RUN wget https://github.com/j-marple-dev/coco-viewer/releases/download/ddd/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb
RUN dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb
RUN apt-get update && apt-get install -y tensorrt && apt-get install -y python3-libnvinfer-dev && rm -rf /var/lib/apt/lists/* 
