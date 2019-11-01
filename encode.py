from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 

import numpy as np
import pandas as pd
from PIL import Image
import glob
import sys
from random import shuffle 

## build keras CNN model
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose 
from keras.layers import Flatten, BatchNormalization, Reshape, Dropout, Input
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras

encoder_file = open("models/encoder_save_10_25_256px_1024_56k_subset.json").read()
Encoder = model_from_json(encoder_file)

## on a new set
v_datagen = ImageDataGenerator(rescale = 1./255)
valid = v_datagen.flow_from_directory("sampled_tiles/training_junk_clusters_removed",
        class_mode="input", batch_size=8, target_size=(256, 256), color_mode = "grayscale",
        shuffle = False, save_to_dir = "transformed/save_transformed_08/", save_prefix = "grey_")

predic_valid = np.asarray(Encoder.predict_generator(valid))
np.savetxt("encodings/56k_training_encodings_1024_10_25.csv", predic_valid, delimiter= ',')

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=6, affinity='euclidean', linkage='ward')
hclust = cluster.fit_predict(predic_valid)

with open("python_h_clusters_56k_6_clust.txt", 'w') as out:
    for i in hclust:
        out.write(str(i) + '\n')

