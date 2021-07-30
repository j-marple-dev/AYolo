import argparse
import json
import os

import pandas as pd

MINOVERLAP = 0.50
BASE_NP = 6554609
BASE_INF = 4690
BASE_AP = 0.29205789973041


def compute_final_score(num_params, inference_time):
    return num_params / BASE_NP + inference_time / BASE_INF


def voc_ap(rec, prec):

    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap, mrec, mpre


def compute_AP(tp, fp, gt_len):

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val
    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val
    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_len
    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap


def cal_IoU(ground_truth_data, dr_data):

    if len(ground_truth_data) == 0:
        if len(dr_data) == 0:
            return [], []
        else:
            return [0] * len(dr_data), [1] * len(dr_data)
    else:
        if len(dr_data) == 0:
            return [], []

    tp, fp = [0] * len(dr_data), [0] * len(dr_data)

    used = [0] * len(ground_truth_data)

    for idx, bb in enumerate(dr_data):
        ovmax = -1
        gt_match = -1

        for jdx, obj in enumerate(ground_truth_data):
            bbgt = obj[:4]
            bi = [
                max(bb[0], bbgt[0]),
                max(bb[1], bbgt[1]),
                min(bb[2], bbgt[2]),
                min(bb[3], bbgt[3]),
            ]
            iw = bi[2] - bi[0] + 1
            ih = bi[3] - bi[1] + 1

            if iw > 0 and ih > 0:
                ua = (
                    (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
                    + (bbgt[2] - bbgt[0] + 1) * (bbgt[3] - bbgt[1] + 1)
                    - iw * ih
                )
                ov = iw * ih / ua

                if ov > ovmax:
                    ovmax = ov
                    gt_match = jdx

        min_overlap = MINOVERLAP
        if ovmax >= min_overlap:
            if gt_match != -1 and used[gt_match] == 0:
                tp[idx] = 1
                used[gt_match] = 1
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1

    return tp, fp


def get_image_annotation(image_fn):

    image_id = image_fn.strip(".jpg")
    video_id = image_fn[:14]
    video_dir = os.path.join(params["data_dir"], "test", video_id)

    with open(os.path.join(video_dir, image_id + ".json")) as json_file:
        data = json.load(json_file)
        image_name = data["file_name"]
        bboxes = data["object"]

        truth = []
        for bbox in bboxes:
            if bbox["label"] == "c_1":
                position = bbox["box"]
                position.append(0)
            truth.append(position)

    return truth


def get_video_annotation(video_dir):
    video_id = video_dir.rsplit("/", 1)[-1]
    with open(os.path.join(video_dir, video_id + ".json")) as json_file:
        annot = json.load(json_file)
    video_annot = {}

    for annot_one_img in annot[video_id]["videos"]["images"]:
        video_annot[annot_one_img["id"]] = annot_one_img["objects"]

    return video_annot  # {'000': [{'id': '00000', 'class_ID': 'c_1', 'position': [...]}, ..., {...}],
    #  '001': [...], ...}


# def get_image_annotation(img_fn: str, data_root: str):
#     video_dir, image_id = img_fn.rsplit('_', 1)
#     image_id = image_id[2:-4]


def get_image_annotation_local(img_fn: str, data_root: str):
    video_dir, image_id = img_fn.rsplit("/", 1)[-1].rsplit("_", 1)
    image_id = image_id[:-4]  # '00000.jpg' > '00000'
    video_annot = get_video_annotation(os.path.join(data_root, video_dir))
    bboxes = video_annot[image_id]

    truth = []
    for bbox in bboxes:
        if (bbox["position"][2] - bbox["position"][0]) < 32 or (
            bbox["position"][3] - bbox["position"][1]
        ) < 32:
            continue

        if bbox["class_ID"] == "c_1":
            position = bbox["position"]
            position.append(0)
        truth.append(position)

    return truth


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    # ap.add_argument('-p', '--project', type=str, default='iitp', help='project file that contains parameters')
    # ap.add_argument('--team_id', type=str, default='1425')
    ap.add_argument("--prediction_file", type=str, help="prediction json file")
    ap.add_argument("--data_root", type=str, help="Directory path for data")
    ap.add_argument("--mock_test", action="store_true")
    ap.add_argument("--wrong_ap", action="store_true")
    args = ap.parse_args()

    # project_name = args.project
    # team_id = args.team_id
    prediction_file = args.prediction_file
    data_root = args.data_root

    # params = yaml.safe_load(open(f'projects/{project_name}.yml'))
    if args.mock_test:
        params = {"data_dir": args.data_root}

    gt_len = 0

    with open(prediction_file, "r") as json_file:
        json_data = json.load(json_file)

    img_annots = json_data["annotations"]
    param_num = float(json_data["param_num"])
    inference_time = float(json_data["inference_time"])

    predictions = []
    for (
        img
    ) in (
        img_annots
    ):  # {"file_name": "...", "objects": [{"position": [...], "confidence_score": "0.342"}, ...]}
        img_fn = img["file_name"]
        img_index = len(predictions)

        img_dt = []
        for i, obj in enumerate(
            img["objects"]
        ):  # obj = {"position": [x1, y1, x2, y2], "confidence_score": "0.826"}
            bbox = {}
            bbox["file_name"] = img_fn
            bbox["obj_id"] = i
            bbox["position"] = obj["position"]
            img_dt.append(obj["position"])
            bbox["confidence_score"] = float(obj["confidence_score"])
            bbox["tp"] = 0
            bbox["fp"] = 0
            predictions.append(bbox)

        if args.mock_test:
            img_gt = get_image_annotation(img_fn)
        else:
            img_gt = get_image_annotation_local(img_fn, data_root)

        tp, fp = cal_IoU(img_gt, img_dt)
        gt_len += len(img_gt)

        for jdx in range(len(img_dt)):
            predictions[img_index + jdx]["tp"] = tp[jdx]
            predictions[img_index + jdx]["fp"] = fp[jdx]

    df_bbox = pd.DataFrame(predictions)
    df_bbox = df_bbox.sort_values(
        by=["confidence_score"], ascending=False, ignore_index=True
    )

    true_positives = list(df_bbox["tp"])
    false_positives = list(df_bbox["fp"])

    ap = compute_AP(true_positives, false_positives, gt_len)
    print("AP:", ap)
    if args.wrong_ap:
        gt_len = len(df_bbox)
        wrong_ap = compute_AP(true_positives, false_positives, gt_len)
        print("wrong_ap: ", wrong_ap)

    print(f"FINAL SCORE: {compute_final_score(param_num, inference_time)}")
