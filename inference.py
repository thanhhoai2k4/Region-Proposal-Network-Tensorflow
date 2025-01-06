import numpy as np
import tensorflow as tf
from Utils import *
import tqdm

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

model.load_weights("weight.weights.h5")

all_anchors = create_anchors_for_feature_map(
        (500, 500),
        (15, 15),
        base_size=16)

image = load_image("data_training/images/maksssksksss101.png",(500,500))
image = np.expand_dims(image, 0)


scores, boxes = model.predict(image)

score = np.reshape(scores, (2025,1))
offset = np.reshape(boxes, (2025,4))

inds = np.where(score > 0.999)[0]

boxes_affter = offset_inverse(all_anchors, offset)

boxes_affter = boxes_affter[inds]
score = score[inds].reshape(len(inds),)
boxes_affter = box_center_to_corner(boxes_affter)
print(score.shape)
print(boxes_affter.shape)

a = tf.image.non_max_suppression(
    boxes_affter,
    score,
        20,
    iou_threshold=0.5,
    score_threshold=float('-inf'),
    name=None
)
a = a.numpy()
boxes_affter = boxes_affter[a]
plot_anchors_xyxy(image[0], boxes_affter)
