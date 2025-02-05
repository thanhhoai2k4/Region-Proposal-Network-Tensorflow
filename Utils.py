import tensorflow as tf
import numpy as np
from PIL import Image
from config import TARGET_SIZE, PATH_IMAGE, PATH_ANNOTATION
import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm



def load_image(image_path: str, target_size: tuple):
    "Load image and resize image to target_size"
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    return img

def parse_Label(xml_file: str, target_size: tuple):
    """
        read xml file and into target size and annotation

        xml_file: path into xml_file
        target_size: want target of image

        return: img, boxes, classes
        shape: (height, width, 3)    (num box, 4)    (num box)
    """
    idx_class = {"without_mask":0, "with_mask":1, "mask_weared_incorrect":2}
    try:
        tree = ET.parse(xml_file)
    except Exception:
        print("Fail")

    root = tree.getroot()

    # width of an image
    width_image = int(root.find("size").find("width").text)
    height_image = int(root.find("size").find("height").text)

    scaleX = float(width_image / target_size[0])
    scaleY = float(height_image / target_size[1])

    image_name = root.find("filename").text
    img = load_image(PATH_IMAGE + image_name, target_size)
    img = np.array(img)/256.0

    boxes = []
    classes = []

    for obj in root.iter("object"):
        cls = obj.find("name").text
        cls = idx_class[cls]
        classes.append(cls)

        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text) / scaleX
        ymin = float(bbox.find("ymin").text) / scaleY
        xmax = float(bbox.find("xmax").text) / scaleX
        ymax = float(bbox.find("ymax").text) / scaleY

        boxes.append(np.array([xmin, ymin, xmax, ymax]))

    return np.array(img), np.array(boxes), np.array(classes)


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = anchors
    c_assigned_bb = assigned_bb
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * np.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = np.concatenate([offset_xy, offset_wh], axis=1)
    return offset


def getXMLs(PATH_ANNOTATION=PATH_ANNOTATION, mode="train"):
    " return list of PATH_ANNOTAION"
    if mode == "train":
        return os.listdir(PATH_ANNOTATION)[0:-100]
    else:
        return os.listdir(PATH_ANNOTATION)[-100:-1]

def Load_data(xml_files=""):
    """
        Load many image
    """
    if xml_files == "":
        xml_files = getXMLs()
    imgs = []
    bbox = []
    category = []
    for i in tqdm(range(len(xml_files))):
        img, boxes, classes = parse_Label(PATH_ANNOTATION + xml_files[i], TARGET_SIZE)
        imgs.append(img)
        bbox.append(boxes)
        category.append(classes)

    bbox = tf.keras.utils.pad_sequences(bbox, padding="post", value=-1)
    # category = tf.keras.utils.pad_sequences(category, padding="post", value=-1)

    return np.array(imgs), bbox,category


def box_corner_to_center(boxes):
    """
    Convert box corners[xmin, ymin, xmax, ymax] to center coordinates[x_center, y_center, width, height].
    boxes : shape[number box, 4]
    return : shape[number box, 4]
    """

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # center_x
    cx = (x1 + x2) / 2
    # center_y
    cy = (y1 + y2) / 2
    # width
    w = (x2 - x1)
    # height
    h = (y2 - y1)
    boxes = np.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """
    apllay for  npArray
    Convert box center coordinates[x_center, y_center, width, height] to corners[xmin, ymin, xmax, ymax].
    boxes: shape[number box, 4]
    return: shape[number box, 4]
    """

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def center_point(imageShape: tuple, fuetureShape: tuple):
    x_center = np.arange(0.5,fuetureShape[0],1)*imageShape[0]//fuetureShape[0]
    y_center = np.arange(0.5, fuetureShape[1], 1)* imageShape[1]//fuetureShape[1]

    center_list = np.meshgrid(x_center, y_center, sparse=False, indexing='xy')
    center_list = np.array(center_list).T.reshape(-1, 2)

    return center_list


def generate_anchors(base_size, ratios=[0.5, 1, 2], scales=[4, 8, 16]):
    """
    Generate anchor boxes based on given base size, aspect ratios, and scales.
    Args:
    - base_size: The base size of the anchor. kich thuoc mat dinh cua 1 anchor
    - ratios: A list of aspect ratios for anchors.
    - scales: A list of scales for anchors.

    Returns:
    - anchors: A list of generated anchor boxes.
    """
    # Initialize an empty list for anchors
    anchors = []

    # Generate anchors for different aspect ratios and scales
    for scale in scales:
        for ratio in ratios:
            # Calculate the width and height for each anchor box
            w = base_size * scale * np.sqrt(ratio)
            h = base_size * scale / np.sqrt(ratio)

            # Append the anchor box (width, height) to the list
            anchors.append((w, h))

    return anchors

def create_anchors_for_feature_map(imageShape: tuple, featueShape:tuple, base_size=16, ratios=[0.5, 1, 2], scales=[4, 8, 16]):
    """
    Create anchor boxes around center points for each feature map cell.
    Args:
    - height: Height of the feature map.
    - width: Width of the feature map.
    - base_size: The base size of the anchor.
    - ratios: List of aspect ratios.
    - scales: List of scales.
    - stride: Stride of the feature map relative to the image.

    Returns:
    - anchors: List of anchor boxes with center points.
    """
    # Generate anchor centers
    centers = center_point(imageShape,featueShape)

    # Generate the base anchor shapes (width, height)
    anchor_shapes = generate_anchors(base_size, ratios, scales)

    all_anchors = []

    # For each center, create anchors
    for (center_y, center_x) in centers:
        for (w, h) in anchor_shapes:
            # Create an anchor with center at (center_y, center_x)
            anchor = np.array([center_x, center_y, w, h])
            all_anchors.append(anchor)

    all_anchors = np.array(all_anchors)

    return np.array(all_anchors)

def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes=boxes.astype(int)
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float32)

    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + 1) * (query_boxes[k, 3] - query_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) - max(boxes[n, 0], query_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) - max(boxes[n, 1], query_boxes[k, 1]) + 1)

                if ih > 0:
                    ua = float((boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua

    return overlaps

def draw_Circle(image, center):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(image)

    for (x,y) in center:
        cirs = plt.Circle((x, y), 10)
        ax.add_patch(cirs)

    plt.show()

def plot_anchors_xywh(image, all_anchors):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    plt.imshow(image)
    for (x,y,w,h) in all_anchors:
        x = x-(w//2)
        y = y-(h//2)
        w = w
        h = h

        rect = Rectangle((x,y), w,h, facecolor='none',edgecolor="red", lw=2 )
        ax.add_patch(rect)

    plt.show()

def plot_anchors_xyxy(image, all_anchors):
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    plt.imshow(image)
    for (xmin, ymin, xmax, ymax) in all_anchors:
        x = xmin
        y = ymin
        w = xmax - xmin
        h = ymax - ymin

        rect = Rectangle((x,y),w,h, facecolor='none',edgecolor="red", lw=2 )
        ax.add_patch(rect)
    plt.show()


def loss_cls(y_true, y_pred):
    y_pred = tf.reshape(y_pred, shape=(-1,2025,1))
    condition = tf.not_equal(y_true, -1)
    indices = tf.where(condition)
    target = tf.gather_nd(y_true, indices)
    output = tf.gather_nd(y_pred, indices)
    loss = tf.keras.losses.BinaryCrossentropy ()
    result = loss(target, output)
    return tf.math.reduce_mean(result)

# https://someshfengde.medium.com/understanding-l1-and-smoothl1loss-f5af0f801c71
def smoothL1( y_true, y_pred, beta=1.0):
    offset_list = y_true[:, :, :-1]
    label_list = y_true[:, :, -1]

    # reshape output by the model
    y_pred = tf.reshape(y_pred, shape=(-1, 2025, 4))
    positive_idxs = tf.where(tf.math.equal(label_list, 1))  # select only foreground boxes

    bbox = tf.gather_nd(y_pred, positive_idxs)
    target_bbox = tf.gather_nd(offset_list, positive_idxs)

    diff = tf.abs(bbox - target_bbox)

    loss = tf.where(
        diff < beta,
        0.5 * tf.square(diff) / beta,
        diff - 0.5 * beta
    )
    return tf.reduce_mean(loss)

def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets.
    apply for NumPy array
    """
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = np.exp(offset_preds[:, 2:] /5) * anc[:, 2:]
    pred_bbox = tf.concat ((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """
    Thực hiện Non-Maximum Suppression (NMS).

    Parameters:
        boxes (ndarray): Mảng các hộp giới hạn (bounding boxes), mỗi hàng là [x1, y1, x2, y2].
        scores (ndarray): Mảng các điểm số (confidence scores) tương ứng với các hộp.
        iou_threshold (float): Ngưỡng IoU để loại bỏ hộp.

    Returns:
        keep (list): Danh sách các chỉ số của các hộp được giữ lại.
    """
    # Kiểm tra nếu không có hộp nào
    if len(boxes) == 0:
        return []

    # Sắp xếp hộp dựa trên điểm số (từ lớn đến nhỏ)
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        # Chọn hộp có điểm số cao nhất
        current = indices[0]
        keep.append(current)

        # Tính IoU giữa hộp hiện tại và các hộp còn lại
        x1 = np.maximum(boxes[current, 0], boxes[indices[1:], 0])
        y1 = np.maximum(boxes[current, 1], boxes[indices[1:], 1])
        x2 = np.minimum(boxes[current, 2], boxes[indices[1:], 2])
        y2 = np.minimum(boxes[current, 3], boxes[indices[1:], 3])

        # Tính diện tích phần giao
        inter_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)
        # Tính diện tích phần union
        box_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        union_area = box_area[current] + box_area[indices[1:]] - inter_area

        # Tính IoU
        iou = inter_area / union_area

        # Loại bỏ các hộp có IoU lớn hơn ngưỡng
        indices = indices[1:][iou <= iou_threshold]

    return keep


# Aplay for TENSORFLOW

def box_center_to_corner_TENSORFLOW(boxes):
    """
    Convert box center coordinates[x_center, y_center, width, height] to corners[xmin, ymin, xmax, ymax].
    boxes: shape[number box, 4]
    return: shape[number box, 4]
    """

    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h

    boxes = tf.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def box_corner_to_center_TENSORFLOW(boxes):
    """
    Convert box corners[xmin, ymin, xmax, ymax] to center coordinates[x_center, y_center, width, height].
    boxes : shape[number box, 4]
    return : shape[number box, 4]
    """

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # center_x
    cx = (x1 + x2) / 2
    # center_y
    cy = (y1 + y2) / 2
    # width
    w = (x2 - x1)
    # height
    h = (y2 - y1)
    boxes = tf.stack((cx, cy, w, h), axis=-1)
    return boxes



def offset_inverse_TENSORFLOW(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets.
    apply for NumPy array
    """
    anc = box_center_to_corner_TENSORFLOW(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = np.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = tf.concat ((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_corner_to_center_TENSORFLOW(pred_bbox)
    return predicted_bbox