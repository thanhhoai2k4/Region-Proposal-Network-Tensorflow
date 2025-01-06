from numpy import ndarray
from tensorflow.python.ops.gen_dataset_ops import model_dataset
from tqdm import tqdm
from Utils import *
import pandas as pd

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





def produce_batch(image: ndarray, gt_boxes: ndarray, outPutShape: tuple):
    """

    :param image: image pixels is loaded it has [width, height, channel]
    :param gt_boxes: ground truth boxes has [x min, y min, x max, y max]
    :param outPutShape: output shape of vgg16 with custom shape
    :return:
        ...
    """

    image_width, image_height = image.shape[:2] # width and height of image roof
    width_feature_map = outPutShape[0] # width of feature == width of output VGG16 with include_top = Fasle
    height_feature_map = outPutShape[1] # ---------------------

    num_feature_map = width_feature_map * height_feature_map # acreage

    # Get all anchors in feature.
    # anchors after creat is [x center, y center, width, height]
    all_anchors = create_anchors_for_feature_map(
        (image_width, image_height),
        (width_feature_map, height_feature_map),
        base_size=16)


    # convert anchors to [x min, y min, x max, y max]
    all_anchors = box_center_to_corner(all_anchors)
    inside_inmage = np.where(
        (all_anchors[:, 0] > 0) &
        (all_anchors[:, 1] > 0) &
        (all_anchors[:, 2] < 500) &
        (all_anchors[:, 3] < 500)
    )[0]

    # Calculate overlaps
    #output shape: number_of_anchors, number_of_ground_trust
    overlaps = bbox_overlaps(all_anchors, gt_boxes)

    argmax_overlaps_rows = overlaps.argmax(axis=1) # get largest in rows : shape[Number_anchors]
    argmax_overlaps_columns = overlaps.argmax(axis=0) # get largest in columns shape[number_gt_box]

    max_overlaps = overlaps[np.arange(len(all_anchors)), argmax_overlaps_rows] # shape: 2025
    gt_max_overlaps = overlaps[argmax_overlaps_columns, np.arange(overlaps.shape[1])] # shape: number_of_gt_box

    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]


    labels = np.empty(shape=(len(all_anchors),1), dtype=np.float32)
    labels.fill(-1)
    labels[gt_argmax_overlaps] = 1
    labels[max_overlaps >= .7] = 1
    labels[max_overlaps <= .3] = 0


    fg_inds = np.where(labels == 1)[0] # tien canh
    num_bg = int(len(fg_inds) * 10) #
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = np.random.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    box_temp = np.empty((len(all_anchors), 4), dtype=np.float32)
    box_temp.fill(-1)
    box_temp = gt_boxes[argmax_overlaps_rows]

    all_anchors = box_corner_to_center(all_anchors)
    box_temp = box_corner_to_center(box_temp)
    offset = offset_boxes(all_anchors, box_temp) # offset of all anchors.
    idpos1 = np.where(labels == -1)[0]
    id0 = np.where(labels == 0)[0]
    offset[idpos1] = -1
    offset[id0] = -1


    labels[inside_inmage] = -1


    offset = np.concatenate((offset, labels), axis=-1)
    return offset, labels, all_anchors

def input_generator(outPutShape = (15,15)):
    print("Load data:")
    images, gt_boxes, classes = Load_data()
    batch_offset = []
    batch_labels = []

    print("Calculate offset labes:")
    for i in tqdm(range((images.shape[0]))):
        imgs = images[i]
        gbs = gt_boxes[i]
        # classes = classes[i]

        offset, labels, all_anchors = produce_batch(imgs, gbs, outPutShape)
        batch_offset.append(offset)
        batch_labels.append(labels)


    batch_offset = np.array(batch_offset)
    batch_labels = np.array(batch_labels)

    return images,batch_offset, batch_labels



images, batch_offset, batch_labels = input_generator()

model.fit(images, [batch_labels, batch_offset], epochs=100, verbose=1)
model.save('RPN.h5')