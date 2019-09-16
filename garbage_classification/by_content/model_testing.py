from keras.models import *
from keras.preprocessing.sequence import pad_sequences
import contents_data_settings
import numpy as np
import pypinyin

model = load_model('/home/alzhu/bluetooth_speaker/train/Bid_model.h5')
model.summary()
generator = {1: '可回收垃圾', 2: '有害垃圾', 4: '湿垃圾', 8: '干垃圾', 16: '大型垃圾'}

while True:
    try:
        print('please input something')
        str = input()
        inputing = []
        for i in str:
            i_pinin = pypinyin.pinyin(i, style=pypinyin.NORMAL)[0][0]
            if i_pinin in Bid_data_setting.name_vocabulary:
                inputing.append(Bid_data_setting.dict_name_num[i_pinin])
        inputing = pad_sequences([inputing], maxlen=Bid_data_setting.pad_num, padding='post')
        prediction = model.predict(inputing)
        prediction = np.argmax(prediction, axis=-1)[0]
        print('the prediction is', generator[prediction])
    except:
        continue
