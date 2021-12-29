import data_utils.face_mask
import tensorflow_datasets as tfds
import cv2

train_ds, val_ds = tfds.load('face_mask', split=['train', 'val'])
for idx, data in enumerate(train_ds):
    if idx > 10:
        break
    print(data)
    image = data['image'].numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    for ymin, xmin, ymax, xmax in data['bboxes']:
        cv2.rectangle(image, (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h)), (0, 255, 0), 3)
    cv2.imshow('frame', image)
    cv2.waitKey(500)
