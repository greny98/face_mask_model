import argparse

import pandas as pd
from tensorflow.keras import optimizers, callbacks

from configs.mask_configs import object_names
from data_utils.mask_generator import MaskGenerator, create_image_info
from model.anchor_boxes import LabelEncoder
from model.losses import RetinaNetLoss
from model.ssd import create_face_mask_model, create_ssd_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--kaggle_dir', type=str, default='data/kaggle_mask')
    parser.add_argument('--medical_dir', type=str, default='data/medical_mask')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='ckpt/face_mask')
    parser.add_argument('--pascal_ckpt', type=str, default='ckpt/pascal/checkpoint')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = vars(parser.parse_args())
    return args


def schedule(e, lr):
    if e < 4 or (e % 4 != 0 and e > 0):
        return lr
    return 0.925 * lr


if __name__ == '__main__':
    args = parse_args()
    print(args)
    info = create_image_info(args['kaggle_dir'], args['medical_dir'])
    train_images = pd.read_csv('train.csv')['filename'].values
    test_images = pd.read_csv('test.csv')['filename'].values
    train_info = {
        filename: info
        for filename, info in info.items() if filename.split('/')[-1] in train_images}
    val_info = {
        filename: info
        for filename, info in info.items() if filename.split('/')[-1] in test_images}

    label_encoder = LabelEncoder()
    train_ds = MaskGenerator(train_info, label_encoder, object_names, training=True, batch_size=args["batch_size"])
    val_ds = MaskGenerator(val_info, label_encoder, object_names, training=False, batch_size=args["batch_size"])
    # Create Model
    ssd_model = create_face_mask_model(args['pascal_ckpt'])
    # ssd_model = create_ssd_model(4)
    loss_fn = RetinaNetLoss(num_classes=4)
    ssd_model.compile(
        loss=loss_fn,
        optimizer=optimizers.Adam(learning_rate=args['lr']))

    ckpt_cb = callbacks.ModelCheckpoint(
        filepath=f"{args['output_dir']}/checkpoint",
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss')
    lr_schedule_cb = callbacks.LearningRateScheduler(schedule)
    # Tensorboard
    tensorboard_cb = callbacks.TensorBoard(log_dir=args['log_dir'])
    ssd_model.fit(train_ds,
                  validation_data=val_ds,
                  callbacks=[ckpt_cb, lr_schedule_cb, tensorboard_cb],
                  epochs=args['epochs'])