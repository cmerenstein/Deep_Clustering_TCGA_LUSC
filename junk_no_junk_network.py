from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
## this is going to be worlds faster with a GPU available
assert tf.test.is_gpu_available()

import numpy as np
import pandas as pd
from PIL import Image
import glob
import sys
from random import shuffle 

from numpy.random import seed
seed(87) #choosing a lucky seed is the most important part

## build keras CNN model
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose 
from keras.layers import Flatten, BatchNormalization, Reshape, Dropout, Input
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.optimizers import Adam

## build keras modelVV
input_image = Input(shape=(256, 256, 1))

model = Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = "same",
    input_shape = (256, 256, 1), kernel_initializer = "he_normal",
    activity_regularizer = l2(0.001))(input_image)
model = MaxPooling2D((2, 2), padding = "same")(model)

model = Conv2D(32, kernel_size = (3,3), activation = 'relu',
        padding = 'same', kernel_initializer = "he_normal",
        activity_regularizer = l2(0.001))(model)
model = (MaxPooling2D((2, 2), padding = "same"))(model)
#
model = (Conv2D(86, kernel_size = (3, 3), activation = 'relu',
    padding = 'same', kernel_initializer = "he_normal",
    activity_regularizer = l2(0.001)))(model)
model = (MaxPooling2D((2, 2), padding = "same"))(model)

model = (Conv2D(86, kernel_size = (3, 3), activation = 'relu',
    padding = 'same', kernel_initializer = "he_normal",
    activity_regularizer = l2(0.001)))(model)
model = (MaxPooling2D((2, 2), padding = "same"))(model)

model = Flatten()(model)
flat1 = Dense(256, activation = 'relu', kernel_initializer = "he_normal",
        activity_regularizer = l2(0.001))(model)
flat2 = Dense(256, activation = 'relu', kernel_initializer = "he_normal",
        activity_regularizer = l2(0.001))(flat1)
model = keras.layers.add([flat1, flat2])

flat3 = Dense(128, activation = 'relu', kernel_initializer = "he_normal",
        activity_regularizer = l2(0.001))(model)

flat4 = Dense(128, activation = 'relu', kernel_initializer = "he_normal",
        activity_regularizer = l2(0.001))(flat3)
model = keras.layers.add([flat3, flat4])

model = Dense(64, activation = 'relu', kernel_initializer = "he_normal",
        activity_regularizer = l2(0.001))(model)

model = Dense(1, activation = 'sigmoid',  kernel_initializer = "he_normal")(model)
model = Model(input_image, model)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics = ['accuracy'])  

## we will have Vto train in batches because the whole thing won't fit in memory
datagen = ImageDataGenerator(featurewise_center=True, rescale = 1./255 )
training = datagen.flow_from_directory("sampled_tiles/junk_no_junk/training/",
        class_mode="binary", batch_size=32, target_size=(256, 256), color_mode = "grayscale" )

valid = datagen.flow_from_directory("sampled_tiles/junk_no_junk/validation", 
        class_mode="binary", batch_size=32, target_size=(256, 256), color_mode = "grayscale")

history = model.fit_generator(training, epochs = 15, validation_data = valid,
        validation_steps = 800)

with open("models/junk_no_junk_10_29_256px_1024_56k_subset.json", 'w') as out:
    out.write(model.to_json())

next_batch = valid.next()
b1 = next_batch[0]
b1_pred = model.predict_on_batch(b1)
q=0
for i in b1:
    img = np.uint8(np.multiply(np.asarray(i), 255))
    img = img.reshape(256, 256)
    Image.fromarray(np.asarray(img)).save("junk_test/test_" + str(q) + "_class_" + 
                        str(next_batch[1][q]) + "_label_" + str(b1_pred[q][0]) +".png")
    q+=1



