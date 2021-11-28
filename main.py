import argparse
from tensorflow.keras import optimizers, callbacks
from datasets.utils import create_image_info, DetectionGenerator
from model.anchor_boxes import LabelEncoder
from model.losses import RetinaNetLoss
from model.ssd import create_ssd_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--kaggle_dir', type=str, default='data/kaggle_mask')
    parser.add_argument('--medical_dir', type=str, default='data/medical_mask')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='model')
    parser.add_argument('--log_dir', type=str, default='logs')
    args = vars(parser.parse_args())
    return args


def schedule(e, lr):
    if e <= 5 and e % 3 != 0:
        return lr
    return 0.975 * lr


if __name__ == '__main__':
    args = parse_args()
    print(args)
    info = create_image_info(args['kaggle_dir'], args['medical_dir'])
    label_encoder = LabelEncoder()
    train_ds, val_ds = DetectionGenerator(info, label_encoder, batch_size=args["batch_size"])
    # Create Model
    ssd_model = create_ssd_model()
    # ssd_model.load_weights("ckpt/classification/checkpoint").expect_partial()
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
