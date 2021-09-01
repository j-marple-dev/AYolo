"""Test dataloader for all available options."""
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expdir",
        type=str,
        default="/usr/src/yolo/runs/exp0",
        help="Run dir ex) runs/exp0/",
    )
    parser.add_argument(
        "--img_size", nargs="+", type=int, default=[480, 480], help="image size"
    )  # height, width
    parser.add_argument("--batch_size", type=int, default=64, help="batchsize")
    parser.add_argument("--num_workers", type=int, default=16, help="num_workers")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset config, if none, load from rundir",
    )
    parser.add_argument(
        "--device", "-d", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument(
        "--dtype",
        "-i",
        type=str,
        default="fp32",
        help="inference type: fp32, fp16, int8",
    )

    opt = parser.parse_args()

    if len(opt.img_size) == 1:
        opt.img_size.append(opt.img_size[-1])

    """
    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp32 --dataloader torch -ie trt'
    print(f"########## [Torch, trt, fp32] ##########")
    print(f"{command}")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp32 --dataloader torch -ie torch'
    print(f"########## [Torch, torch, fp32] ##########")
    print(f"{command}")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp16 --dataloader torch -ie trt'
    print(f"########## [Torch, trt, fp16] ##########")
    print(f"{command}")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp16 --dataloader torch -ie torch'
    print(f"########## [Torch, torch, fp16] ##########")
    print(f"{command}")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp32 --dataloader dali -ie trt'
    print(f"########## [Dali, trt, fp32] ##########")
    print(f"{command}")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp32 --dataloader dali -ie torch'
    print(f"########## [Dali, torch, fp32] ##########")
    print(f"{command}")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp16 --dataloader dali -ie trt'
    print(f"########## [Dali, trt, fp16] ##########")
    print(f"{command}")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp16 --dataloader dali -ie torch'
    print(f"########## [Dali, torch, fp16] ##########")
    print(f"{command}")
    os.system(command)
    """
    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp16 --dataloader dali -ie torch'
    print("########## [Dali, torch, fp16] ##########")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp32 --dataloader dali -ie trt'
    print("########## [Dali, trt, fp32] ##########")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype fp16 --dataloader dali -ie trt'
    print("########## [Dali, trt, fp16] ##########")
    os.system(command)

    command = f'export PYTHONPATH="$PWD" && python benchmark/dataloader_test/run_loader.py --expdir {opt.expdir} --img_size {opt.img_size[0]} {opt.img_size[1]} --batch_size {opt.batch_size} --num_workers {opt.num_workers} --device {opt.device} --dtype int8 --dataloader dali -ie trt'
    print("########## [Dali, trt, int8] ##########")
    os.system(command)
