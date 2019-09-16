from keras.models import *
from keras.utils import to_categorical
import cv2 as cv
import numpy as np
import settings
import data_setting

model = load_model('Inception_Resnet_v2_Model.h5')
model.summary()

inputing, outputing = [], []
row = np.random.randint(0, len(data_setting.img_names), settings.batch_size)
img_names = data_setting.img_names[row]
img_labels = data_setting.img_labels[row]
img_names_labels = {}
for i in range(len(img_names)):
    img_names_labels[img_names[i]] = img_labels[i]
for path, label in img_names_labels.items():
    img = cv.imread(data_setting.images_file_path + path)
    img = cv.resize(img, (settings.img_rows, settings.img_cols))
    img = np.array(img)
    img = img / 255
    inputing.append(img)
    outputing.append(label)
inputing = np.array(inputing)
outputing = np.array(outputing)
outputing = to_categorical(outputing, len(data_setting.objects_name_num))
print(model.evaluate(inputing, outputing, settings.batch_size))

while True:
    print('please input your path')
    img_path = input()
    img = cv.imread(img_path)
    img = cv.resize(img, (settings.img_rows, settings.img_cols))
    img = np.array([img])
    img = img / 255
    prediction = model.predict(img)
    prediction_num = np.argmax(prediction, axis=-1)
    print('the prediction_num is', data_setting.objects_num_name[str(prediction_num[0])])
