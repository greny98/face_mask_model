import os
import json
from datasets.values import NAME_TO_LABEL
from os.path import join


def read_medical_mask(annotation, images_dir, target: dict):
    for filename in os.listdir(annotation):
        if '.json' not in filename:
            continue
        with open(join(annotation, filename)) as fp:
            info = json.load(fp)
            image_info = {
                "bboxes": [],
                "labels": []
            }
            for anno in info["Annotations"]:
                name = anno["classname"]
                label = NAME_TO_LABEL.get(name)
                if label is not None:
                    image_info["bboxes"].append(anno["BoundingBox"])
                    image_info["labels"].append(label)
        target[join(images_dir, info["FileName"])] = image_info
    return target
