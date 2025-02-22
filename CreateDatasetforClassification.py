from Utils import create_anchors_for_feature_map, parse_Label, bbox_overlaps, plot_anchors_xyxy, plot_anchors_xywh, box_corner_to_center, box_center_to_corner
import os
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import cm

class_id = ["without_mask", "with_mask","mask_weared_incorrect"]
path_images = "data_training/images/"
path_annot = "data_training/annotations/"
target_size = (500,500)

xmls =  sorted(
    os.path.join(path_annot, file_name)
        for file_name in os.listdir(path_annot)
        if file_name.endswith(".xml")
)


def creatDataset():
    if (os.path.isdir("data_training_classification") == False):
        os.makedirs("data_training_classification")
        if (os.path.isdir("data_training_classification/images") == False):
            os.makedirs("data_training_classification/images")
        os.makedirs("data_training_classification/images/without_mask")
        os.makedirs("data_training_classification/images/with_mask")
        os.makedirs("data_training_classification/images/mask_weared_incorrect")
        os.makedirs("data_training_classification/images/background")


    # anchors
    all_anchors = create_anchors_for_feature_map((500, 500), (15, 15))
    all_anchors = box_center_to_corner(all_anchors) # corner xyxy

    for xml in xmls:
        # get the image, box, class
        image, boxes, classes = parse_Label(xml, target_size) # xyxy

        # iou
        overlaps = bbox_overlaps(all_anchors, boxes)

        inds_rows = overlaps.argmax(axis=1) # get max in rows
        max_values = overlaps[np.arange(len(all_anchors)),inds_rows]

        inds_values_background = np.where(
            ( max_values < 0.4 ) &
            ( max_values > 0.1 )
        )[0]

        box_background = all_anchors[inds_values_background][:5] # holding five boxes
        box_foreground = boxes



        # cut boxes from an image
        for box in box_background:
            # Xác định tọa độ cắt (trái, trên, phải, dưới)
            left = int( box[0] if box[0] >0 else 0)
            top = int( box[1] if box[1] >0 else 0)
            right = int( box[2] if box[2] < 500 else 500)
            bottom =int( box[3] if box[3] < 500 else 500)

            # Cắt ảnh
            cropped_array = image[top:bottom, left:right, :] * 255
            cropped_array = cropped_array.astype(np.uint8)

            im = Image.fromarray(cropped_array)
            im = im.resize((100,100))
            im.save("data_training_classification/images/background/" + str(len(os.listdir("data_training_classification/images/background")) + 1) + ".jpg"  )

        i = 0
        for box in box_foreground:

            # Xác định tọa độ cắt (trái, trên, phải, dưới)
            left = int(box[0] if box[0] > 0 else 0)
            top = int(box[1] if box[1] > 0 else 0)
            right = int(box[2] if box[2] < 500 else 500)
            bottom = int(box[3] if box[3] < 500 else 500)

            # Cắt ảnh
            cropped_array = image[top:bottom, left:right, :] * 255
            cropped_array = cropped_array.astype(np.uint8)

            im = Image.fromarray(cropped_array)
            im = im.resize((100, 100))

            if classes[i] == 0: # without_mask
                im.save("data_training_classification/images/without_mask/" + str(len(os.listdir("data_training_classification/images/without_mask")) + 1) + ".jpg")
            elif classes[i] == 1: # with_mask
                im.save("data_training_classification/images/with_mask/" + str(len(os.listdir("data_training_classification/images/with_mask")) + 1) + ".jpg")
            else: # mask_weared_incorrect
                im.save("data_training_classification/images/mask_weared_incorrect/" + str(len(os.listdir("data_training_classification/images/mask_weared_incorrect")) + 1) + ".jpg")
            i += 1

creatDataset()