import numpy as np
from config import *
from Utils import *
from config import TARGET_SIZE

# model base Region proposal network
def modelRPN():
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

    model = tf.keras.Model(inputs=[vgg16.input], outputs=[output_scores, output_deltas])
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=0.001,  # Tốc độ học
        momentum=0.9,  # Giá trị Momentum
        nesterov=True  # Có sử dụng Nesterov Momentum hay không
    )
    model.compile(optimizer=optimizer, loss={'scores1':loss_cls, 'deltas1':smoothL1})
    return model

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
    N = len(xmls)
    i = 0

    outPutShape = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(500, 500, 3),
    ).layers[-1].output.shape[1:3]  # 15,15
    i = 0
    for xml in xmls:
        img, gt_boxes = parse_Label11(xml, TARGET_SIZE)
        offset, lb, all_anchors = produce_batch(img, gt_boxes, outPutShape)
        yield img, (lb, offset)

# build model
model = modelRPN()
# load dataset
dataset = tf.data.Dataset.from_generator(
    lambda: getData("train"),
    output_signature=(
        tf.TensorSpec(shape=(500,500,3), dtype=tf.float32),  # Đầu vào X_batch image
        (
            tf.TensorSpec(shape=(2025, 1), dtype=tf.float32),# Đầu ra 1 (hồi quy) score
            tf.TensorSpec(shape=(2025,5), dtype=tf.float32),# Đầu ra 2 (phân loại) offset
        )
    )
)

data_val = tf.data.Dataset.from_generator(
    lambda: getData("val"),
    output_signature=(
        tf.TensorSpec(shape=(500,500,3), dtype=tf.float32),  # Đầu vào X_batch image
        (
            tf.TensorSpec(shape=(2025, 1), dtype=tf.float32),# Đầu ra 1 (hồi quy) score
            tf.TensorSpec(shape=(2025,5), dtype=tf.float32),# Đầu ra 2 (phân loại) offset
        )
    )
)


lenght = len(getXMLs())
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="weight.weights.h5",  # Đường dẫn lưu mô hình
    monitor="loss",       # Tiêu chí giám sát (vd: val_loss, val_accuracy, loss, etc.)
    save_best_only=True,      # Chỉ lưu mô hình tốt nhất
    save_weights_only=True,  # Lưu toàn bộ mô hình (False) hoặc chỉ weights (True)
    mode="min",               # 'min' (val_loss thấp nhất) hoặc 'max' (accuracy cao nhất)
    verbose=1         # In log khi lưu mô hình
)


dataset = dataset.shuffle(SHUFFLE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

data_val = data_val.shuffle(SHUFFLE)
data_val = data_val.batch(BATCH_SIZE)
data_val = data_val.prefetch(tf.data.AUTOTUNE)

model.fit(dataset,epochs=50,verbose=1, steps_per_epoch=lenght//BATCH_SIZE, callbacks=[model_checkpoint_callback], validation_data=data_val)