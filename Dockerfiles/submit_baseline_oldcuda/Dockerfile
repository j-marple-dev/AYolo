FROM nvcr.io/nvidia/tensorrt:19.10-py3

WORKDIR /dep 
COPY . . 
RUN apt-get update && apt-get install -y libgl1-mesa-dev && apt-get -y install jq && ./download_trt.sh && pip install --upgrade pip && pip install -r requirements.txt && pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda100 && rm -rf /var/lib/apt/lists/* &&  rm -rf requirements.txt 
