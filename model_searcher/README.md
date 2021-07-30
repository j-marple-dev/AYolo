# Auto Model Search with Optuna

Searching hyper-parameter and model structures with [Optuna](https://optuna.org).

# 1. How to run 

## 1.1. Environments

- python > 3.7
- `pip install -r requirements.txt` 


## 1.2. optimize_main.py usage examples
### 1.2.1. Basic Run
  - `python optimize_main.py -d ${DATASET_YAML} --img-size 640 --epochs 50 --single-cls -tnp 1.5 -map-t 0.3 --device 0 --storage "postgresql://your_id:your_password@server-address.com/optuna" --study-name ${study_name}`
  

## 1.3. Option Details
  - `-tnp`, `--threshold-n-param`: Threshold of parameter number from baseline parameter.
    - Number of parameter must be smaller than (${BASELINE_N_PARAM} * ${THRESHOLD_N_PARAM}). Otherwise, it prunes the trial.
  - `-map-t`, `--map-threshold`: Threshold of mAP@0.5 from baseline.
    - mAP@0.5 on test set must be over  (${BASELINE_mAP} * ${mAP_THRESHOLD}). Otherwise, it prunes the trial.
  - `-noo`, `--no-optimize-option`: No optimization of training options.
  - `-noh`, `--no-optimize-hyp`:  No optimization of hyper-parameters.
  - `-noa`, `--no-optimize-aug`:  No optimization of augmentation.
  - `-nsm`, `--no-search-model`:  No searching for the model.
  - `--cfg`: Use fixed model if provided and ${no_search_model} flag is on.
  - `--study-name`: Within same `${study-name}` on same `${storage}`, they share optimization results.
  - `--storage`: Database server address.
  - `--wlog`: Logging on wandb.
  - `--wlog-tags`: Add tags for the experiments.


```shell
usage: optimize_main.py [-h] [-d DATA] [-nt N_TRIALS] [-ts TEST_STEP]
                        [-w WORKERS] [-is IMG_SIZE [IMG_SIZE ...]] [-noo]
                        [-noh] [-noa] [-nsm] [--cfg CFG]
                        [-tnp THRESHOLD_N_PARAM] [-map-t MAP_THRESHOLD]
                        [--device DEVICE] [--epochs EPOCHS] [--no-prune]
                        [--batch-size BATCH_SIZE] [--single-cls]
                        [--logdir LOGDIR] [--name NAME]
                        [--study-name STUDY_NAME] [--storage STORAGE]

Searching for the best model

optional arguments:
  -h, --help            show this help message and exit
  -d DATA, --data DATA  Dataset yaml file (default: data/aigc_local.yaml)
  -nt N_TRIALS, --n-trials N_TRIALS
                        Number of trials for the searching (default: 1000)
  -ts TEST_STEP, --test-step TEST_STEP
                        Perform test on every ${test_step} epochs (default: 1)
  -w WORKERS, --workers WORKERS
                        maximum number of dataloader workers (default: 8)
  -is IMG_SIZE [IMG_SIZE ...], --img-size IMG_SIZE [IMG_SIZE ...]
                        [train, test] image sizes (default: [640, 640])
  -noo, --no-optimize-option
                        No optimization of training options. (default: False)
  -noh, --no-optimize-hyp
                        No optimization of hyper-parameters. (default: False)
  -noa, --no-optimize-aug
                        No optimization of augmentation. (default: False)
  -nsm, --no-search-model
                        No searching for the model. (default: False)
  --cfg CFG             Use fixed model if provided and ${no_search_model}
                        flag is on. (default: )
  -tnp THRESHOLD_N_PARAM, --threshold-n-param THRESHOLD_N_PARAM
                        Threshold of parameter number from baseline parameter.
                        (default: 0.8)
  -map-t MAP_THRESHOLD, --map-threshold MAP_THRESHOLD
                        Threshold of mAP@0.5 from baseline. (default: 0.95)
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu (default: )
  --epochs EPOCHS       Number of epochs on each trial. (default: 30)
  --no-prune            No prune model while training. (default: False)
  --batch-size BATCH_SIZE
                        total batch size for all GPUs (default: 32)
  --single-cls          train as single-class dataset (default: False)
  --logdir LOGDIR       logging directory (default: runs/)
  --name NAME           renames experiment folder exp{N} to exp{N}_{name} if
                        supplied (default: )
  --study-name STUDY_NAME
                        Name of the study for Optuna (default: aigc_model)
  --storage STORAGE     Optuna storage URL (default: )

```


## 1.3. optuna_optimizer/optuna_watcher.py usage examples
### 1.2.1. Basic Run
  - `python optuna_optimizer/optuna_watcher.py "postgresql://your_id:your_password@server-address.com/optuna" --study-name ${study_name}`
  
```shell
$ python optuna_optimizer/optuna_watcher.py "postgresql://your_id:your_password@server-address.com/optuna" --study-name yolo_only_anchor --attrs worker mAP0_5 n_param                   

Loading ...
0edfe5b59022: 4, 15c4e97e14b8: 11, 40dbc57f039d: 32, 5c6244079531: 4, 6ebd440a0c19: 11, 7f9c58ed1d1a: 7, afc03fe1719d: 5, b7a32ae783d8: 4, cbc8bedbe520: 46, jeikei-desktop: 9, jeikei-ubuntu-pc: 64, jongkuk-J: 46, nipa2019-0250: 29
Total: 301
  STATE  |Trial No|  Score  |  |                                    User Attributes                 |               Start date    
COMPLETE |  2,880 | 2.12852 |  | mAP0_5: 0.598508122234455 | n_param: 497610 | worker: afc03fe1719d | 2020-1107-18:12:30 (runtime - 03:52:02)
COMPLETE |  3,140 | 2.14957 |  | mAP0_5: 0.6302162664242502 | n_param: 733258 | worker: 0edfe5b59022 | 2020-1108-01:35:01 (runtime - 04:11:10)
COMPLETE |  3,432 | 2.15135 |  | mAP0_5: 0.6149803311282418 | n_param: 512082 | worker: nipa2019-0250 | 2020-1108-15:41:20 (runtime - 00:48:14)
COMPLETE |  2,193 | 2.17417 |  | mAP0_5: 0.6231799331323573 | n_param: 440538 | worker: jeikei-ubuntu-pc | 2020-1107-17:09:39 (runtime - 00:31:31)
COMPLETE |  3,184 | 2.21564 |  | mAP0_5: 0.6324305040465057 | n_param: 585658 | worker: jeikei-ubuntu-pc | 2020-1108-08:30:53 (runtime - 00:31:33)
COMPLETE |  3,342 | 2.23863 |  | mAP0_5: 0.6265061130608196 | n_param: 595874 | worker: jeikei-ubuntu-pc | 2020-1108-12:33:57 (runtime - 00:31:18)
COMPLETE |  2,475 | 2.25991 |  | mAP0_5: 0.5995784274297289 | n_param: 456258 | worker: jeikei-ubuntu-pc | 2020-1107-19:33:07 (runtime - 00:32:33)
COMPLETE |  2,217 | 2.26310 |  | mAP0_5: 0.6165387081505378 | n_param: 522690 | worker: jeikei-ubuntu-pc | 2020-1107-17:46:26 (runtime - 00:31:45)
COMPLETE |  3,370 | 2.26353 |  | mAP0_5: 0.6200222573185147 | n_param: 595874 | worker: jeikei-ubuntu-pc | 2020-1108-13:05:11 (runtime - 00:31:14)
COMPLETE |  3,297 | 2.26383 |  | mAP0_5: 0.6169066956087634 | n_param: 577370 | worker: jeikei-ubuntu-pc | 2020-1108-11:29:57 (runtime - 00:31:26)
COMPLETE |  3,240 | 2.27137 |  | mAP0_5: 0.6253125528668849 | n_param: 665394 | worker: jeikei-ubuntu-pc | 2020-1108-10:12:59 (runtime - 00:31:35)
COMPLETE |  3,319 | 2.27139 |  | mAP0_5: 0.6299325806836185 | n_param: 677890 | worker: jeikei-ubuntu-pc | 2020-1108-12:01:27 (runtime - 00:31:28)
COMPLETE |  3,078 | 2.27376 |  | mAP0_5: 0.6128163284170414 | n_param: 760242 | worker: afc03fe1719d | 2020-1107-23:35:21 (runtime - 03:48:10)
COMPLETE |  2,697 | 2.27566 |  | mAP0_5: 0.6139588820414641 | n_param: 716978 | worker: b7a32ae783d8 | 2020-1107-16:30:15 (runtime - 04:22:27)
COMPLETE |  2,948 | 2.27876 |  | mAP0_5: 0.6122280377190079 | n_param: 572722 | worker: jeikei-ubuntu-pc | 2020-1108-02:13:45 (runtime - 00:31:08)

```
  
## 1.3. Option Details
  - `--ls`: List all study names and summary.
  - `n-top N_TOP`: Shows `${N_TOP}` number of trials.
  - `--query QUERY`: Show a detailed trial result. 
  - `--rm`: Remove `${STUDY_NAME}` study. (Re-type of the study name required)
  - `--force-rm`: Remove `${STUDY_NAME}` study. Remove the study immediately.
  - `--show-prune`: Include pruned trials to show.
  - `--params [PARAMS [PARAMS ...]]`: Show parameter values.
  - `--attrs  [ATTRS [ATTRS ...]]`: Show user attribute values.
  - `--importance`: Show the parameter importance. `sklearn` required.
  - `--sort-attr SORT_ATTR`: Sort by `${SORT_ATTR}` instead of objective score value.
  - `--overwrite-attr`: Overwrite study attributes with `${STUDY_CONF}`. 
  - `--info`: Show study information.

  
  
```shell
usage: optuna_watcher.py [-h] [--study-name STUDY_NAME] [--verbose VERBOSE]
                         [--n-top N_TOP] [--ls] [--rm] [--force-rm]
                         [--show-prune] [--params [PARAMS [PARAMS ...]]]
                         [--attrs [ATTRS [ATTRS ...]]] [--importance]
                         [--query QUERY] [--sort-attr SORT_ATTR] [--sort-date]
                         [--create] [--direction DIRECTION]
                         [--study-conf STUDY_CONF] [--overwrite-attr] [--info]
                         storage

Optuna CLI.

positional arguments:
  storage               postgresql://your_id:your_password@server_address.com/
                        optuna

optional arguments:
  -h, --help            show this help message and exit
  --study-name STUDY_NAME
                        Study name to query. (default: )
  --verbose VERBOSE     Verbosity level (default: 1)
  --n-top N_TOP         Number of top results to show (default: 20)
  --ls                  Show list of studies in storage (default: False)
  --rm                  Remove ${study-name} (default: False)
  --force-rm            Force remove ${study-name} (default: False)
  --show-prune          Include pruned results (default: False)
  --params [PARAMS [PARAMS ...]]
                        Parameter names to display (Multiple) (default: None)
  --attrs [ATTRS [ATTRS ...]]
                        User attribute names to display (Multiple) (default:
                        None)
  --importance          Show parameter importance of the study (default:
                        False)
  --query QUERY         Query index to display detailed parameters and user
                        attributes (default: -1)
  --sort-attr SORT_ATTR
                        Sort by user attribute values. (default: )
  --sort-date           Sort by recent studies (default: False)
  --create              Create study named ${study-name} (default: False)
  --direction DIRECTION
                        Objective score direction. (minimize or maximize
                        (default: minimize)
  --study-conf STUDY_CONF
                        Study configuration yaml file path. (default:
                        optuna_optimizer/config/study_conf.yaml)
  --overwrite-attr      Overwrite study user attributes. (default: False)
  --info                Show study information. (default: False)

```
