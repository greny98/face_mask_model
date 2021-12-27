from tensorflow.keras import layers, Model, regularizers

from configs.common_config import EXTEND_CONV_FIlTER
from configs.pascal_configs import PASCAL_LABELS
from model.feature_pyramid import get_backbone, FeaturePyramid, BackboneLarge, FeaturePyramidLarge

l2 = regularizers.l2(1.5e-5)


def build_head(feature, name):
    for i in range(4):
        feature = layers.DepthwiseConv2D(3, padding="same", name=name + '_depthwise' + str(i))(feature)
        feature = layers.Conv2D(EXTEND_CONV_FIlTER, 1, padding="same", name=name + '_conv' + str(i))(feature)
        feature = layers.BatchNormalization(epsilon=1.001e-5, name=f'{name}_bn_{i}')(feature)
        feature = layers.ReLU(name=f"{name}_relu_{i}")(feature)
    return feature


def build_head_large(feature, name):
    for i in range(4):
        feature = layers.Conv2D(EXTEND_CONV_FIlTER, 3, padding="same", name=name + '_conv' + str(i))(feature)
        feature = layers.BatchNormalization(epsilon=1.001e-5, name=f'{name}_bn_{i}')(feature)
        feature = layers.ReLU(name=f"{name}_relu_{i}")(feature)
    return feature


def ssd_head(features):
    classes_outs = []
    box_outputs = []
    for idx, feature in enumerate(features):
        classify_head = build_head(feature, 'classify_head' + str(idx))
        detect_head = build_head(feature, 'detect_head' + str(idx))
        box_outputs.append(detect_head)
        classes_outs.append(classify_head)
    return classes_outs, box_outputs


def create_ssd_model(num_classes, large=False):
    if large:
        backbone = BackboneLarge()
        pyramid = FeaturePyramidLarge(backbone, filters=128)
    else:
        backbone = get_backbone()
        pyramid = FeaturePyramid(backbone)
    classes_heads, box_heads = ssd_head(pyramid.outputs)

    num_anchor_boxes = 9
    classes_outs = []
    box_outputs = []
    for idx, head in enumerate(box_heads):
        detect_head = layers.Conv2D(num_anchor_boxes * 4, 1, padding="same",
                                    name='detect_head' + str(idx) + '_conv_out',
                                    kernel_regularizer=l2)(head)
        box_outputs.append(layers.Reshape([-1, 4], name='detect_reshape_' + str(idx))(detect_head))

    for idx, head in enumerate(classes_heads):
        classify_head = layers.Conv2D(num_anchor_boxes * num_classes, 1, padding="same",
                                      name='classify_head' + str(idx) + '_conv_out',
                                      kernel_regularizer=l2)(head)
        classes_outs.append(layers.Reshape([-1, num_classes])(classify_head))

    classes_outs = layers.Concatenate(axis=1)(classes_outs)
    box_outputs = layers.Concatenate(axis=1, name="concat_box_head")(box_outputs)
    outputs = layers.Concatenate(axis=-1)([box_outputs, classes_outs])
    return Model(inputs=[backbone.input], outputs=[outputs])


def create_face_mask_model(pascal_ckpt):
    pascal_model = create_ssd_model(len(PASCAL_LABELS))
    pascal_model.load_weights(pascal_ckpt).expect_partial()
    box_outputs = pascal_model.get_layer("concat_box_head").output
    classes_heads = [l.output for l in pascal_model.layers if (f"_relu_3" in l.name) and 'classify_head' in l.name]

    num_classes = 4
    num_anchor_boxes = 9
    classes_outs = []
    for idx, head in enumerate(classes_heads):
        classify_head = layers.Conv2D(num_anchor_boxes * num_classes, 1, padding="same",
                                      name='classify_head' + str(idx) + '_conv_out',
                                      kernel_regularizer=l2)(head)
        classes_outs.append(layers.Reshape([-1, num_classes])(classify_head))
    classes_outs = layers.Concatenate(axis=1)(classes_outs)
    outputs = layers.Concatenate(axis=-1)([box_outputs, classes_outs])
    return Model(inputs=[pascal_model.input], outputs=[outputs])
