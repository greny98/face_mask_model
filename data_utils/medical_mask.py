import os
import json
from os.path import join

from configs.mask_configs import NAME_TO_LABEL


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
            if len(image_info["labels"]) > 0:
                target[join(images_dir, info["FileName"])] = image_info
    return target
