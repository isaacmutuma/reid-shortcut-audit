import os
import cv2
import numpy as np
train_dir = os.path.join(config.DATASET_ROOT, 'market1501', 'Market-1501-v15.09.15', 'bounding_box_train')

'''     This file given the training directory,
        load each image with OpenCV,
        crop the torso,
        extract dominant hue,
        and return a dict of {image_path: color_label}.
'''



def get_dominant_hue(img_hsv):
    # compute histogram of H channel
    h_channel = img_hsv[:, :, 0]  # shape (100, 128)
    counts, edges = np.histogram(h_channel, bins=18, range=(0, 180))
    # find the bin with most pixels
    dominant_bin = np.argmax(counts)
    # return the dominant hue range
    dominant_hue = edges[dominant_bin]
    # mean saturation
    mean_saturation = img_hsv[:, :, 1].mean()
    return dominant_hue, mean_saturation


def hue_to_label(dominant_hue, mean_saturation):
    # if saturation is low → achromatic (black/white/gray) → label 0
    if mean_saturation < 50:
        return 0  # achromatic — black, white, gray
    elif dominant_hue <= 10 or dominant_hue >= 170:
        return 1  # red
    elif dominant_hue <= 25:
        return 2  # orange
    elif dominant_hue <= 35:
        return 3  # yellow
    elif dominant_hue <= 85:
        return 4  # green
    elif dominant_hue <= 130:
        return 5  # blue
    else:
        return 6  # purple


def extract_color_labels(train_dir):
    color_labels = {}

    for filename in sorted(os.listdir(train_dir)):
        if filename.endswith('.jpg'):
            # build full path
            full_path = os.path.join(train_dir, filename)
            # load image with cv2
            img = cv2.imread(full_path)
            # crop torso rows 70-170
            img_torso = img[70:170, :]
            # convert to HSV
            img_hsv = cv2.cvtColor(img_torso, cv2.COLOR_BGR2HSV)
            # get dominant hue
            dominant_hue, mean_saturation = get_dominant_hue(img_hsv)
            # get color label
            label = hue_to_label(dominant_hue, mean_saturation)
            # store in dict with full path as key
            color_labels[full_path] = label

    return color_labels
