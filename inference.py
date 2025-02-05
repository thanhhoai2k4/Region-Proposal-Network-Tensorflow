import numpy as np
import tensorflow as tf
from Utils import *
import tqdm
def RPN():
    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(500, 500, 3),
    )

    outPutShape = vgg16.layers[-1].output.shape[1:3] # 15,15

    k=9 #anchor number for each point
    #################  RPN Model  #######################
    convolution_3x3 = tf.keras.layers.Conv2D(
        filters=512,
        kernel_size=(3, 3),
        padding='same',
        name="3x3"
    )(vgg16.output)

    output_deltas = tf.keras.layers.Conv2D(
        filters= 4 * k,
        kernel_size=(1, 1),
        activation="linear",
        kernel_initializer="uniform",
        name="deltas1"
    )(convolution_3x3)

    output_scores = tf.keras.layers.Conv2D(
        filters=1 * k,
        kernel_size=(1, 1),
        activation="sigmoid",
        kernel_initializer="uniform",
        name="scores1"
    )(convolution_3x3)

    model = tf.keras.Model(inputs=[vgg16.input], outputs=[output_scores, output_deltas])
    model.compile(optimizer='adam', loss={'scores1':loss_cls, 'deltas1':smoothL1})

    model.load_weights("gate/weight.weights.h5")
    return model


def main(imagepath, model):
    # xywh
    all_anchors = create_anchors_for_feature_map(
            (500, 500),
            (15, 15),
            base_size=16)
    all_anchors = box_center_to_corner(all_anchors)
    # load image
    image = load_image(imagepath,(500,500))
    image1 = np.asarray(image)/255.0
    image1 = np.expand_dims(image1, 0)
    # Predict
    scores, offset = model.predict(image1)

    # Reshape
    scores = np.reshape(scores, (2025,1))
    offset = np.reshape(offset, (2025,4))

    inds = np.where(scores > 0.5)[0]

    # convert offset into xyxy
    boxes = offset_inverse(all_anchors, offset)

    boxes = boxes[inds]
    scores = scores[inds].reshape(len(inds),)


    # inside_IM = np.where(
    #     (boxes[:, 0] > 0)&
    #     (boxes[:, 1] > 0)&
    #     (boxes[:, 2] < 500)&
    #     (boxes[:, 3] < 500)
    # )
    # boxes = boxes[inside_IM]
    # scores = scores[inside_IM]

    # apply NMS.
    keep = tf.image.non_max_suppression(
        boxes,
        scores,
            20,
        iou_threshold=0.5,
        score_threshold=float('-inf'),
        name=None).numpy()
    boxes = boxes[keep]

    plot_anchors_xyxy(image, boxes)


main("Image_Test/04.jpg", RPN())