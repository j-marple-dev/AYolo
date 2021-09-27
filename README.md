# Auto YOLO
This repository is based on [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).

## TODO
- [*] Training w/ w/o wandb
- [ ] Test with local torch weights
- [ ] AutoML with optuna optimizer


# Environment setup
## Prerequisites
* python 3.7
* [Anaconda](https://www.anaconda.com/distribution/)
* Docker (Optional)
* Clone this repository
```bash
$ https://github.com/j-marple-dev/AYolo.git
$ cd AYolo
```

## Installation
### 1. Create virtual environment and install requirements
```bash
$ conda create -n AYolo python=3.7
$ conda activate AYolo
$ pip install -r requirements.txt
```

### 2. (Optional for nvidia gpu) Install cudatoolkit.
```bash
$ conda activate AYolo
$ conda install -c pytorch cudatoolkit=${cuda_version}
```

### 3. (Optional for wandb logging) Setting wandb login
```bash
$ wandb login
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter: {PASTE WANDB API KEY}
```

## Using Docker
### 1. Docker Build
```bash
$ sudo docker build --tag {your_docker:tag} .
```

## 1.2 Docker Run
```bash
# restart docker system control
$ sudo systemctl restart docker

# (Optionoal) Plotting images on the screen
$ xhost +

# Run docker image
$ sudo docker run --ipc=host --gpus 0 -it -v {dataset_dir}:/usr/src/data/ -v {yolo_repo_dir}:/usr/src/yolo -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY {your_docker:tag} /bin/bash
```

# 2. Training

* Train model with config and save the model in local or wandb.
## model config yaml

model config yaml(eg. models/yolov5s.yaml) looks like :
```yaml
nc: num_class
depth_multiple: depth_multiple  # model depth multiplier
width_multiple: width_ultiple   # model width multiplier

anchors:
  - [x1, y1, x2, y2, x3, y3]  # P3
  - [x4, y4, x5, y5, x6, y6]  # P4
  - [x7, y7, x8, y8, x9, y9]  # P5

backbone:
  # [from, iterations, module, args]
  [[-1, 1, Focus, [64, 3]],  #  Focus layer exmaple
   [-1, 1, Conv, [128, 3, 2]],  # Convolution layer example
   ...
  ]

head:
  [[-1, 1, Conv, [512, 1, 1]]  # Convolution example
   ...
  ]
```

## Train usage
```bash
python train.py --help

usage: train.py [-h] [--weights WEIGHTS] [--cfg CFG] [--data DATA] [--hyp HYP]
                [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--img-size IMG_SIZE [IMG_SIZE ...]] [--rect]
                [--resume [RESUME]] [--nosave] [--notest] [--noautoanchor]
                [--evolve] [--bucket BUCKET] [--cache-images]
                [--cache-images-multiprocess] [--image-weights] [--name NAME]
                [--device DEVICE] [--multi-scale] [--single-cls] [--adam]
                [--sync-bn] [--local_rank LOCAL_RANK] [--logdir LOGDIR]
                [--workers WORKERS] [--no-weight-wandb] [--wlog]
                [--wlog-project WLOG_PROJECT] [--check-git-status]
                [--test_every_epoch TEST_EVERY_EPOCH] [--n-skip N_SKIP]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     initial weights path (default: )
  --cfg CFG             model.yaml path or wandb path (default: )
  --data DATA           data.yaml path (default: data/coco128.yaml)
  --hyp HYP             hyperparameters path (default: data/hyp.scratch.yaml)
  --epochs EPOCHS
  --batch-size BATCH_SIZE
                        total batch size for all GPUs (default: 16)
  --img-size IMG_SIZE [IMG_SIZE ...]
                        [train, test] image sizes (default: [640, 640])
  --rect                rectangular training (default: False)
  --resume [RESUME]     resume most recent training (default: False)
  --nosave              only save final checkpoint (default: False)
  --notest              only test final epoch (default: False)
  --noautoanchor        disable autoanchor check (default: False)
  --evolve              evolve hyperparameters (default: False)
  --bucket BUCKET       gsutil bucket (default: )
  --cache-images        cache images for faster training (default: False)
  --cache-images-multiprocess
                        cache images with multi-cores (default: False)
  --image-weights       use weighted image selection for training (default:
                        False)
  --name NAME           renames experiment folder exp{N} to exp{N}_{name} if
                        supplied (default: )
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu (default: )
  --multi-scale         vary img-size +/- 50% (default: False)
  --single-cls          train as single-class dataset (default: False)
  --adam                use torch.optim.Adam() optimizer (default: False)
  --sync-bn             use SyncBatchNorm, only available in DDP mode
                        (default: False)
  --local_rank LOCAL_RANK
                        DDP parameter, do not modify (default: -1)
  --logdir LOGDIR       logging directory (default: runs/)
  --workers WORKERS     maximum number of dataloader workers (default: 8)
  --no-weight-wandb     Skip loading weights from wandb model. (default:
                        False)
  --wlog                Use wandb to log training status (default: False)
  --wlog-project WLOG_PROJECT
                        Wandb project name (default: ayolo_train)
  --check-git-status    Check git status if the branch is behind from the
                        remote. (default: False)
  --test_every_epoch TEST_EVERY_EPOCH
  --n-skip N_SKIP       Skip every n data on dataset. (default: 0)
```

# 3. Prediction

## 3.1 Predict with trained model
Compare the experiment model with the baseline model: inference time, total parameters, evaluation metrics(precision, recall, mAP, etc).

#### Data config
```yaml
# train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
train: dataset_root/train/
val: dataset_root/val/

# number of classes
nc: 20

# class names
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

#### Running usage
```bash
$ python test.py --help
usage: test.py [-h] [--expdir EXPDIR] [--basedir BASEDIR] [--data DATA]
               [--batch BATCH] [--conf-thres CONF_THRES]
               [--iou-thres IOU_THRES] [--device DEVICE] [--single-cls]
               [--augment] [--verbose] [--wlog]
               [--profile-iteration PROFILE_ITERATION] [--plots]
               [--plot-criterion {None,large_conf_small_iou,small_conf_large_iou}]
               [--no-small-box] [--no-small-box-infer]

Compare experiment model with baseline model

optional arguments:
  -h, --help            show this help message and exit
  --expdir EXPDIR, -e EXPDIR
                        Experiment model path or Wandb run path, ex)
                        runs/{experiment_name}/
  --basedir BASEDIR, -b BASEDIR
                        Baseline model, to get relative score depending on the
                        machine
  --data DATA           Test data config, ex) data/aigc.yaml default: Same as
                        experiment model
  --batch BATCH, -bs BATCH
                        Test batchsize
  --conf-thres CONF_THRES, -ct CONF_THRES
                        object confidence threshold
  --iou-thres IOU_THRES, -it IOU_THRES
                        IOU threshold for NMS
  --device DEVICE, -d DEVICE
                        cuda device, i.e. 0 or 0,1,2,3 or cpu
  --single-cls, -sc     treat as single-class dataset
  --augment, -a         augmented inference
  --verbose, -v         report mAP by class
  --wlog, -w            Enable wandb
  --profile-iteration PROFILE_ITERATION, -pi PROFILE_ITERATION
                        Profiling iteration number.
  --plots               show plots (on wandb when it is used)
  --plot-criterion {None,large_conf_small_iou,small_conf_large_iou}
                        Criterion for filtering plots
  --no-small-box        Filter out small bboxes in ground-truth labels
  --no-small-box-infer  Filter out small bboxes in the inferenced

$ python test.py -e ${wandb_runpath} -b ${baseline_wandb_runpath} --data ./data/aigc.yaml  # basic run
$ python test.py -e ${wandb_runpath} -b ${baseline_wandb_runpath} --data ./data/aigc.yaml --wlog --plots  # plotting bboxes on wandb
``` 



## 3.2 Predict with converted model
#### Config yaml file
```yaml
Dataset:
  annot_yolo: false
  batch_size: 16
  data_root: /usr/src/data/aigc_format  # data root
  img_size: 480  # img size after preprocessing (before calculate rectangular or padding)
  load_json: false
  original_shape:
    - 1080 # height
    - 1920 # width
  pad: 0.5  # add padding
  preload: false  # preload images or not
  preload_multiprocess: true  # preload images using multiprocess or not
  rect: true  # use rectangular image or not
  stride: 16  # ratio (original_img / result of last neck layer)
  videos:  # test video names
    - test
    - video
    - names
    - ...
conf_thres: 0.1  # confidence threshold using during nms
dataloader: dali  # data loader (dali or torch)
device: '0'  # gpu device
dtype: fp32  # data type of weights (fp32 fp16 int8)
iou_thres: 0.6  # IoU threshold
model: torch  # model (torch if torchscript, trt if tensorrt)
padded_img_size:  # padded image size
  - 288
  - 496
path: export/{your/export/path}  # path with converted models and config file
rank: -1  
workers: 8  # dataloader workers
```
#### Running usage

```bash
# ${config_file_path}: trt_config.yaml path.
# ${test_dataset_dir}: test image directory.
$ python tensorrt_run/predict.py --config ${config_file_path} --data-dir ${test_dataset_dir}

usage: predict.py [-h] --config CONFIG --data-dir DATA_DIR

optional arguments:
  -h, --help           show this help message and exit
  --config CONFIG      config file path. (default: None)
  --data-dir DATA_DIR  test image dir. (default: /usr/src/data/)
```

# 4. Converting

### Converting pipeline

1. Convert pytorch model to onnx model  ([onnx](https://onnx.ai))
2. Convert onnx model to TensorRT model ([TensorRT](https://developer.nvidia.com/Tensorrt))

## 4.1 Convert pytorch model to onnx and torchscript

```bash
# ${weights_file}: torch weight file path(.pt) or wandb path
$ python models/torch_to_onnx.py \
--weights ${weights_file} --img-size ${input_img_size} --batch-size ${batch_size} \
--config ${base/config/file/path.yaml} --rect --pad ${pad} --dataloader ${dali_or_torch} --conf_thres ${confidence_threshold} \
--torchscript  # store true

usage: torch_to_onnx.py [-h] [--weights WEIGHTS] [--img-size IMG_SIZE [IMG_SIZE ...]] [--batch-size BATCH_SIZE] [--torchscript] [--coreml]
                        [--config CONFIG] [--download_root DOWNLOAD_ROOT] [--export_root EXPORT_ROOT] [--rect] [--pad PAD]
                        [--dataloader DATALOADER] [--conf_thres CONF_THRES]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     weights path (default: ./yolov5s.pt)
  --img-size IMG_SIZE [IMG_SIZE ...]
                        image size height, width (default: [480, 480])
  --batch-size BATCH_SIZE
                        batch size (default: 16)
  --torchscript         Export a model to TorchScript (default: False)
  --coreml              Export a model to CoreML (default: False)
  --config CONFIG       base config file. (default: config/base_config.yaml)
  --download_root DOWNLOAD_ROOT
                        wandb download root. (default: wandb/downloads)
  --export_root EXPORT_ROOT
                        export root for onnx and torchscript file. (default: export)
  --rect                rectangular image or not (store, true) (default: False)
  --pad PAD             pad ratio(stride), applied only when rect is true (default: 0.5)
  --dataloader DATALOADER
                        Data loader type (dali/ torch) (default: dali)
  --conf_thres CONF_THRES
                        confidence threshold for NMS (default: 0.1)
```

## 4.2 Convert onnx model to tensorrt model

```bash
$ python models/onnx_to_trt.py ${expdir} --dtype ${data_type(fp16 or int8)} --device 0 --calib-imgs ${calib_imags_dir} --gpu-mem ${available_gpu_memory(GBs)}

usage: onnx_to_trt.py [-h] [--dtype DTYPE] [--profile-iter PROFILE_ITER]
                      [--top-k TOP_K] [--keep-top-k KEEP_TOP_K]
                      [--torch-model TORCH_MODEL] [--device DEVICE]
                      [--calib-imgs CALIB_IMGS] [--gpu-mem GPU_MEM]
                      expdir

positional arguments:
  expdir                experiment dir. ex) export/runs/run_name/

optional arguments:
  -h, --help            show this help message and exit
  --dtype DTYPE         datatype: fp32, fp16, int8 (default: fp16)
  --profile-iter PROFILE_ITER
                        Number profiling iteration. (default: 100)
  --top-k TOP_K         Top k number of NMS in GPU. (default: 512)
  --keep-top-k KEEP_TOP_K
                        Top k number of NMS in GPU. (default: 100)
  --torch-model TORCH_MODEL
                        Torch model path. Run profiling if provided. (default:
                        )
  --device DEVICE       GPU device for PyTorch. (default: )
  --calib-imgs CALIB_IMGS
                        image directory for int8 calibration. (default:
                        /usr/src/trt/yolov5/build/calib_imgs)
  --gpu-mem GPU_MEM     Available GPU memory. (default: 8)
```

# 5. AutoML using Optuna
* Find the lightweight object detection model while maintaining mAP using [Optuna](https://optuna.org).

## 5.1. Prerequisites
* RDB server (Optional but strongly recommend)
  * We use [postgresql](https://www.postgresql.org) but Optuna supports multiple types of RDBs(MySQL, Oracle, Microsoft SQL, and SQLite). Please refer to [SQLAlchemy document](https://docs.sqlalchemy.org/en/14/core/engines.html#database-urls) and [Optuna RDB backend document](https://optuna.readthedocs.io/en/stable/tutorial/003_rdb.html).
* wandb (Optional but strongly recommend)
  * [wandb(Weights & biases)](https://wandb.ai/) is strong platform to manage model trainings. We utilize wandb to manage AutoML search process.


## 5.2. Setting
* model_searcher/config/**study_conf.yaml**
```yaml
direction: minimize
study_attr:
  baseline_path: null  # Use wandb path if necessary.
  target_param: 100000  # Target parameter.
  target_mAP: 0.8  # Target mAP@0.5 score.
  mAP_weight: 4.0  # Weight value for mAP@0.5 score
  no_prune_epoch: 10  # Optuna will not prune the study with in no_prune_epoch
  optimize:
    opt: True  # Search for the option parameters. (adam, image_weights, multi_scale training)
    hyp: False  # Search for the hyper-parameters.
    augment: False  # Search for the augmentations.
    model: True  # Search for the models.
hyp_config: model_searcher/config/hyp_config.yaml  # Hyper-parameter and augmentation options
```

* model_searcher/config/**hyp_config.yaml**
```yaml
param:
  lr0:
    default: 0.01
    suggest:
      range: [0.005, 0.05]  # [min, max]
      n_step: 5  # Number of splits between [min, max] -> [0.005, 0.01625, 0.0275, 0.03875, 0.05]
  lrf:
    default: 0.2
    suggest:
      range: []  # Default value will be used if range is empty.
      n_step: 5
      ...
      ...
      ...
augment:
  hsv_h:
    default: 0.015
    suggest:
      range: [0.0, 0.3]
      n_step: 5
      ...
      ...
      ...
```

## 5.3. Objective Score
* Objective score: <img src="https://render.githubusercontent.com/render/math?math=\color{gray}arg\min%20{%20(\omega \cdot (1-mAP) %2B t %2B p)}" />
  * ω: mAP_weight
  * mAP: mAP@0.5 score
  * t: time score(Mean inference time for a single image) - <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\frac{t_{target}}{t_{baseline}}" />
  * p: parameter score - <img src="https://render.githubusercontent.com/render/math?math=\color{gray}\frac{p_{target}}{p_{baseline}}" />


## 5.4. Usage
* Example
```bash
$ python optimize_main.py --data data/coco.yaml --epochs 50 --batch-size 8 --no-single-cls --study-name {STUDY_NAME} --storage postgresql://{ACCOUNT}:{PASSWORD}@{SERVER_ADDRESS}/optuna --wlog-project {WANDB_PROJECT}
```

* Arguments
```bash
$ python optimize_main.py --help
usage: optimize_main.py [-h] [-d DATA] [-nt N_TRIALS] [-ts TEST_STEP]
                        [-w WORKERS] [-is IMG_SIZE [IMG_SIZE ...]] [-noo]
                        [-noh] [-noa] [-nsm] [--override-optimization]
                        [--cfg CFG] [-tnp THRESHOLD_N_PARAM] [--device DEVICE]
                        [--epochs EPOCHS] [--no-prune]
                        [--batch-size BATCH_SIZE] [--no-single-cls]
                        [--single-cls] [--logdir LOGDIR] [--name NAME]
                        [--study-name STUDY_NAME] [--study-conf STUDY_CONF]
                        [--storage STORAGE] [--wlog]
                        [--wlog-project WLOG_PROJECT]
                        [--wlog-tags [WLOG_TAGS [WLOG_TAGS ...]]] [--no-cache]
```

## 5.5. Update study configurations
* The updated configurations will be applied without interrupting runs.
```bash
$ python optuna_watcher.py postgresql://{ACCOUNT}:{PASSWORD}@{SERVER_ADDRESS}/optuna --study-name {STUDY_NAME} --overwrite-attr model_searcher/config/study_conf.yaml
```
