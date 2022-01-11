import data_utils.fire_ds
import tensorflow_datasets as tfds
import cv2

train_ds, val_ds, test_ds = tfds.load('fire_ds', split=['train', 'val', 'test'])
for idx, data in enumerate(val_ds):
    if idx > 20:
        break
    image = data['image'].numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    for ymin, xmin, ymax, xmax in data['bboxes']:
        cv2.rectangle(image, (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h)), (0, 255, 0), 3)
    cv2.imshow('frame', image)
    cv2.waitKey(0)
