import keras
import numpy as np
from config import *
from Utils import *
from config import TARGET_SIZE

class OffsetToBox(tf.keras.layers.Layer):
    def __init__(self,anchors, **kwargs):
        super(OffsetToBox, self).__init__(**kwargs)
        self.anchors = anchors
    def call(self, offsets):
        offsets = tf.reshape(offsets,(-1,2025,4))
        dx = tf.reshape(offsets[...,0], (-1,1))
        dy = tf.reshape(offsets[...,1], (-1,1))
        dw = tf.reshape(offsets[...,2], (-1,1))
        dh = tf.reshape(offsets[...,3], (-1,1))

        x_a = tf.reshape(self.anchors[...,0], (-1,1))
        y_a = tf.reshape(self.anchors[...,1], (-1,1))
        w_a = tf.reshape(self.anchors[...,2], (-1,1))
        h_a = tf.reshape(self.anchors[...,3], (-1,1))

        x_center = (dx/10) * w_a + x_a
        y_center = (dy/10) * h_a + y_a
        width = tf.exp(dw/5) * w_a
        height = tf.exp(dh/5) *h_a

        x_center = tf.reshape(x_center, (-1,))
        y_center = tf.reshape(y_center, (-1,))
        width = tf.reshape(width, (-1,))
        height = tf.reshape(height, (-1,))
        boxes = tf.stack([x_center, y_center, width, height], axis=1)

        return boxes

    def compute_output_shape(self, input_shape):
        # Assuming the input shapes are (batch_size, num_anchors, 4) for both anchors and offsets
        return [(input_shape[0], input_shape[1], 4)]  # (batch_size, num_anchors, 4) for boxes


class NMS_1(tf.keras.layers.Layer):
    def __init__(self, keep = 20, iou_threshold = 0.5, **kwargs):
        super(NMS_1, self).__init__(**kwargs)
        self.keep = keep
        self.iou_threshold = iou_threshold

    def call(self, boxes, scores):
        pass

    def compute_output_shape(self, input_shape):
        """
            keep nms
        """
        return [(input_shape[0], self.keep, 4)]

# model base Region proposal network
def Faster_Rcnn(image_width, image_height, width_feature_map, height_feature_map):
    """"""
    all_anchors = create_anchors_for_feature_map(
        (image_width, image_height),
        (width_feature_map, height_feature_map),
        base_size=16)
    all_anchors = tf.convert_to_tensor(all_anchors, dtype=tf.float32)
    """"""
    # Load VGG16 with it's weights.
    vgg16 = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(500, 500, 3),
    )

    # unfreeze some last CNN layer:
    for layer in vgg16.layers:
        layer.trainable = False

    # Get output shape in feature
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

    model1 = tf.keras.Model(inputs=[vgg16.input], outputs=[output_scores, output_deltas])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.001,  # Tốc độ học
        momentum=0.9,  # Giá trị Momentum
        nesterov=True  # Có sử dụng Nesterov Momentum hay không
    )
    model1.compile(optimizer=optimizer, loss={'scores1':loss_cls, 'deltas1':smoothL1})
    model1.load_weights("gate/weight.weights.h5")

    box = OffsetToBox(all_anchors)(output_deltas)
    mns = NMS_1()(box, output_scores)
    model2 = tf.keras.Model(inputs=[vgg16.input], outputs=[output_scores, box])

    return model1, model2

def produce_batch(image, gt_boxes, outPutShape:tuple):
    image_width, image_height = image.shape[:2]  # width and height of image roof
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

    offset = np.concatenate((offset, labels), axis=-1)
    return offset, labels, all_anchors

def getData(mode):
    xmls = getXMLs(mode=mode)
    for i in range(len(xmls)):
        xmls[i] = PATH_ANNOTATION + xmls[i]

    outPutShape = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(500, 500, 3),
    ).layers[-1].output.shape[1:3]  # 15,15
    for xml in xmls:
        img, gt_boxes, classes = parse_Label(xml, TARGET_SIZE)
        offset, lb, all_anchors = produce_batch(img, gt_boxes, outPutShape)
        yield img, (lb, offset)


# build model
model1, model2 = Faster_Rcnn(500,500,15,15)




a = load_image("Image_Test/10.png", (500,500))
b = np.asarray(a)/255.0
a1 = np.expand_dims(b,0)


scores, box = model2.predict(a1)
box = box[0]
scores = np.reshape(scores,(2025,1))
idx = np.where(scores>0.7)[0]
box = box[idx]
plot_anchors_xywh(a, box)
