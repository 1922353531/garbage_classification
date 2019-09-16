from keras.utils import to_categorical
import cv2 as cv
import numpy as np
import json
import re
import os
import glob
import settings
import random

# labels
labels_file_path = 'datasets/garbage_classify/garbage_classify_rule.json'
with open(labels_file_path, encoding='UTF-8') as json_file:
    file = json.load(json_file)
objects_num_name = dict([(i, char) for i, char in file.items()])
objects_name_num = dict([(char, i) for i, char in objects_num_name.items()])

# images
img_names, img_labels = [], []
images_file_path = 'datasets/garbage_classify/train_data/'
images_files = glob.glob(images_file_path + '*')
for file in images_files:
    if file[-1] == 't':
        with open(file, encoding='UTF-8') as f:
            f = list(f)[0]
            f_result = re.search('(.*?),\s(.*?)$', f)
            img_name = f_result.group(1)
            img_label = int(f_result.group(2))
            img_names.append(img_name)
            img_labels.append(img_label)
img_names = np.array(img_names)
img_labels = np.array(img_labels)
