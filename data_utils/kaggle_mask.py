import os
from os.path import join
from configs.mask_configs import KAGGLE_MASK_LABELS

from data_utils.pascal import xml_to_dict


def read_kaggle_mask(annotation: str, images_dir: str, target: dict) -> dict:
    for xml_file in os.listdir(annotation):
        if ".xml" not in xml_file:
            continue
        anno = xml_to_dict(join(annotation, xml_file))
        image_info = {
            "bboxes": [],
            "labels": [],
            'width': anno["size"]["width"],
            'height': anno["size"]["height"]
        }
        for obj in anno["object"]:
            bbox = obj["bndbox"]
            image_info["bboxes"].append([
                bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]
            ])
            image_info["labels"].append(KAGGLE_MASK_LABELS[obj["name"]])
            image_info['from'] = 'kaggle'
        target[join(images_dir, anno["filename"])] = image_info
    return target
