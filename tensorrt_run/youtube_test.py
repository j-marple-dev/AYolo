import argparse

import cv2
import numpy as np
import torch
import yaml
import youtube_dl

from tensorrt_run.predict import load_model
from utils.general import non_max_suppression
from utils.torch_utils import select_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loader_config", type=str, default="model/submit_config.yaml")
    parser.add_argument("--device", "-d", default="0", help="cuda device")
    parser.add_argument(
        "--url", default="https://www.youtube.com/watch?v=Svwk36ENSns", type=str
    )
    parser.add_argument("--iou-t", default=0.6, type=float)
    parser.add_argument("--conf-t", default=0.1, type=float)
    parser.add_argument("--view-scale", default=2.0, type=float)
    opt = parser.parse_args()

    download_path = f"./tmp/{opt.url.split('=')[-1]}.mp4"

    ydl_opts = {
        "format": "best/best",
        "outtmpl": download_path,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([opt.url])

    device = select_device(opt.device)

    with open(opt.loader_config) as f:
        config = yaml.load(f, yaml.FullLoader)

    cap = cv2.VideoCapture(download_path)
    success, image = cap.read()

    h, w = image.shape[:2]

    model = load_model(
        config["path"],
        # config["model"],
        "torch",
        config["dtype"],
        config["dataloader"],
        config["Dataset"]["batch_size"],
        device,
    )

    r = config["Dataset"]["img_size"] / 1920
    if config["Dataset"]["rect"]:
        img_h, img_w = (
            np.ceil(
                np.array((1080 * r, 1920 * r)) / config["Dataset"]["stride"]
                + config["Dataset"]["pad"]
            )
            * config["Dataset"]["stride"]
        )
    else:
        img_h, img_w = (1920 * r,) * 2

    img_h, img_w = int(img_h), int(img_w)

    print(f"Image size: ({img_h}, {img_w})")

    while success:
        image = cv2.resize(image, (img_w, img_h))
        torch_image = (
            torch.Tensor(image).permute(2, 0, 1).unsqueeze(dim=0).half().to(device)
            / 255.0
        )

        result = model(torch_image)
        nms_result = non_max_suppression(result[0], opt.conf_t, opt.iou_t)[0]

        if nms_result is not None:
            for bbox in nms_result.cpu().detach().numpy():
                cv2.rectangle(
                    image,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 0, int(bbox[4] * 255)),
                    thickness=max(int(3 * bbox[4]), 1),
                )
                cv2.putText(
                    image,
                    f"{bbox[4]:.2f}",
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (int(bbox[4] * 255), 0, 0),
                    1,
                )

        image = cv2.resize(
            image, (int(img_w * opt.view_scale), int(img_h * opt.view_scale))
        )
        cv2.imshow("a", image)
        cv2.waitKey(1)
        success, image = cap.read()
