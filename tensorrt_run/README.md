# 1. Docker Run
  - `xhost +`
  - `sudo docker run --ipc=host --gpus all -it -v ${dataset_root}:/usr/src/data/ -v ${repo_dir}:/usr/src/yolo -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY jmarpledev/submit:v0.2 /bin/bash`

 ```bash
 # Test dataloader
 python src/dataset/dataset_dali.py
 # run export.py first and make model, weight ready
 python predict.py
 ```

# 2. Evaluate json files

```bash
python evaluate.py --prediction_file ${result_json_filepath} --data_root ${data_root}
```
where `result_json_filepath` is the filepath of the json filepath of "predicted" bboxes, and `data_root` is the root directory whose structure is as follows:
```
data_root/
  +- 0506_V0001/
    +- image/
      +- xxx.jpg
  :
  +- 0615_V0047/
    +- image/
      +- xxx.jpg
```

```bash
python evaluate.py --prediction_file ${result_json_filepath} --data_root ${data_root} --mock_test
```
where `result_json_filepath` is the filepath of the json filepath of "predicted" bboxes, and `data_root` is the root directory whose structure is as follows:
```
data_root/
  +- test/
    +- t1_video_00000/
      +- xxx.jpg
      :
    :
    +- t1_video_00469/
      +- xxx.jpg
```
