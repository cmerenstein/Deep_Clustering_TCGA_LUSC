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

encoder = Conv2D(8, kernel_size = (3,3), activation = 'relu', padding = "same",
    input_shape = (256, 256, 1), kernel_initializer = "he_normal")(input_image)
encoder = MaxPooling2D((2, 2), padding = "same")(encoder)

encoder = Conv2D(16, kernel_size = (3,3), activation = 'relu',
        padding = 'same', kernel_initializer = "he_normal")(encoder)
encoder = (MaxPooling2D((2, 2), padding = "same"))(encoder)
#
encoder = (Conv2D(32, kernel_size = (3, 3), activation = 'relu',
    padding = 'same', kernel_initializer = "he_normal"))(encoder)
encoder = (MaxPooling2D((2, 2), padding = "same"))(encoder)

encoder = (Conv2D(64, kernel_size = (3, 3), activation = 'relu',
    padding = 'same', kernel_initializer = "he_normal"))(encoder)
encoder = (MaxPooling2D((2, 2), padding = "same"))(encoder)

#encoder = (Conv2D(128, kernel_size = (3, 3), activation = 'relu',
#    padding = 'same', kernel_initializer = "he_normal"))(encoder)
#encoder = (MaxPooling2D((2, 2), padding = "same"))(encoder)
#
#encoder = (Conv2D(128, kernel_size = (3, 3), activation = 'relu',
#    padding = 'same', kernel_initializer = "he_normal"))(encoder)
#
encoder = Flatten()(encoder)
flat1 = Dense(1024, activation = 'tanh', kernel_initializer = "he_normal",
        activity_regularizer = l2(0.001))(encoder)
flat2 = Dense(1024, activation = 'tanh', kernel_initializer = "he_normal",
        activity_regularizer = l2(0.001))(flat1)
encoder = keras.layers.add([flat1, flat2])

flat3 = Dense(1024, activation = 'tanh', kernel_initializer = "he_normal",
        activity_regularizer = l2(0.001))(encoder)
model = keras.layers.concatenate([encoder, flat3])
model = Reshape((16, 16, 8))(model)
#
#model = (Conv2D(128, kernel_size = (3, 3), activation = 'relu', 
#    padding = 'same', kernel_initializer = "he_normal"))(model) 
#model = (UpSampling2D((2, 2)))(model)
#
#model = (Conv2D(128, kernel_size = (3, 3), activation = 'relu',
#    padding = 'same', kernel_initializer = "he_normal"))(encoder) 
#model = (UpSampling2D((2, 2)))(model)

model = (Conv2D(64, kernel_size = (3, 3), activation = 'relu',
    padding = 'same', kernel_initializer = "he_normal"))(model) 
model = (UpSampling2D((2, 2)))(model)

model = (Conv2D(32, kernel_size = (3, 3), activation = 'relu', 
    padding = 'same', kernel_initializer = "he_normal"))(model)
model = (UpSampling2D((2, 2)))(model)

model = (Conv2D(16, kernel_size = (3, 3), activation = 'relu',
    padding = "same", kernel_initializer = "he_normal"))(model)
model = (UpSampling2D((2, 2)))(model)

model = Conv2D(8, kernel_size = (3,3), activation = 'relu',
        padding = "same", kernel_initializer = "he_normal")(model)
model = (UpSampling2D((2, 2)))(model)

model = Conv2D(1, (3, 3), activation = "sigmoid", padding = "same")(model)

model = Model(input_image, model)
model.compile(optimizer=Adam(lr=0.0001), loss='mse')  
Encoder = Model(input_image, encoder)

## we will have Vto train in batches because the whole thing won't fit in memory
datagen = ImageDataGenerator(featurewise_center=True, rescale = 1./255, validation_split=0.05)
training = datagen.flow_from_directory("sampled_tiles/test_256_var_filt/", class_mode="input",
        batch_size=8, target_size=(256, 256), color_mode = "grayscale"      )
history = model.fit_generator(training, epochs = 50)

with open("models/model_save_10_15_256px_1024_100k.json", 'w') as out:
    out.write(model.to_json())

with open("models/encoder_save_10_15_256px_1024_100k.json", 'w') as out:
    out.write(Encoder.to_json())

b1_lumin = training.next()[0]
q=0
for i in np.asarray(b1_lumin): 
    img = np.uint8(np.multiply(np.asarray(i), 255))
    img = img.reshape(256, 256)
    Image.fromarray(np.asarray(img)).save("test_" + str(q) + ".png")
    q += 1

q=0
for i in model.predict_on_batch(b1_lumin):
    print(i.shape)
    img = np.uint8(np.multiply(np.asarray(i), 255))
    img = img.reshape(256, 256)
    Image.fromarray(img).save("pred_" + str(q) + ".png")
    q+=1

## on a new set
v_datagen = ImageDataGenerator(rescale = 1./255, featurewise_center=True)
valid = v_datagen.flow_from_directory("sampled_tiles/larger_valid_256_var_filt", class_mode="input",
        batch_size=8, target_size=(256, 256), color_mode = "grayscale",
        shuffle = False, save_to_dir = "transformed/save_transformed_04/", save_prefix = "grey_")

b1_lumin = valid.next()[0]
q=5
for i in np.asarray(b1_lumin): 
    img = np.uint8(np.multiply(np.asarray(i), 255))
    img = img.reshape(256, 256)
    Image.fromarray(np.asarray(img)).save("test_" + str(q) + ".png")
    q += 1

q=5
for i in model.predict_on_batch(b1_lumin):
    print(i.shape)
    img = np.uint8(np.multiply(np.asarray(i), 255))
    img = img.reshape(256, 256)
    Image.fromarray(img).save("pred_" + str(q) + ".png")
    q+=1

predic_valid = np.asarray(Encoder.predict_generator(valid))
np.savetxt("encodings/validation_encoding_1024_centered_var_filt.csv", predic_valid, delimiter= ',')


