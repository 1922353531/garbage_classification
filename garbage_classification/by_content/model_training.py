from keras.layers import *
from keras.models import *

import contents_data_settings
import settings

inputing = Input(shape=(Bid_data_setting.pad_num, ))
embedding = Embedding(len(Bid_data_setting.name_vocabulary), settings.embedding_dim, input_length=Bid_data_setting.pad_num, trainable=True)(inputing)
bid = Bidirectional(LSTM(settings.LSTM_neurons, return_sequences=True), merge_mode='concat', trainable=True)(embedding)
conv1d_1 = Conv1D(settings.filters_1, kernel_size=settings.kernel_size, padding='valid', activation='relu')(bid)
conv1d_2 = Conv1D(settings.filters_2, kernel_size=settings.kernel_size, padding='valid', activation='relu')(conv1d_1)
maxpooling1d_1 = MaxPooling1D()(conv1d_2)
flatten = Flatten()(maxpooling1d_1)
dense_1 = Dense(512, activation='tanh')(flatten)
dense_2 = Dense(17, activation='softmax')(dense_1)
outputing = dense_2
model = Model(inputing, outputing)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
print('the summary of the model is')
model.summary()


model.fit(Bid_data_setting.input_datas, Bid_data_setting.target_datas, epochs=settings.epochs, steps_per_epoch=settings.step_per_epoch)
model.save('/home/alzhu/bluetooth_speaker/train/Bid_model.h5')
print('the model has been saved.')

model.save('/home/alzhu/bluetooth_speaker/train/Bid_model.h5')
print('the model has been saved.')



