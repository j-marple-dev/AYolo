FROM nvcr.io/nvidia/tensorrt:20.03-py3

WORKDIR /dep 
COPY . . 
RUN apt-get update && apt-get install -y libgl1-mesa-dev && apt-get -y install jq && ./download_trt.sh && pip install --upgrade pip && pip install -r requirements.txt && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100 && dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb && apt-get update && apt-get install -y tensorrt libcudnn8 && apt-get install -y python3-libnvinfer-dev && rm -rf /var/lib/apt/lists/* &&  rm -rf download_trt.sh nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.1.6-ga-20201006_1-1_amd64.deb requirements.txt 
