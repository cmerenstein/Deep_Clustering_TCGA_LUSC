from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf 
## this is going to be worlds faster with a GPU available
assert tf.test.is_gpu_available()

import numpy as np
from PIL import Image

from numpy.random import seed
seed(87) #choosing a lucky seed is the most important part

## build keras CNN model
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose 
from keras.layers import Flatten, BatchNormalization, Reshape, Dropout, Input
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Layer, InputSpec
import keras
from keras.optimizers import Adam, SGD
import keras.backend as K
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans

from functions_for_keras_clustering_autoencoder import build_model, ClusteringLayer

## get a fresh model and the pretrained version for initializing the K means
encoder, autoencoder = build_model()
pretrained_encoder = model_from_json(open("models/encoder_save_10_15_256px_1024_100k.json").read())

## we need a lot of clusters, most will be small, some should be large
n_clusters = 50

## build the combined model
clustering_layer = ClusteringLayer(n_clusters, name="clustering")(encoder.output)
model = Model(inputs = encoder.input, outputs = [clustering_layer, autoencoder.output])
model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=Adam(lr=0.0001))

## training on 13k images, 22 images from each slide. This could be increased
datagen = ImageDataGenerator(featurewise_center=True, rescale = 1./255)
training = datagen.flow_from_directory("sampled_tiles/larger_valid_256_var_filt/", class_mode="input",
        batch_size=8, target_size=(256, 256), color_mode = "grayscale"      )

## predict the initial K means clusters on the training set
kmeans = KMeans(n_clusters=n_clusters, n_init = 20)
y_pred = kmeans.fit_predict(pretrained_encoder.predict_generator(training))
y_pred_last = np.copy(y_pred)

#initialize centers
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q, _ = model.predict_generator(training)
        p = target_distribution(q)  # update the auxiliary target distribution p
        y_pred = q.argmax(1)
#
        # check stop criterion - model convergence
        print("here")
	delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 2000 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    print(ite)
    idx = index_array[index * batch_size: min( (index + 1) * batch_size, y_pred.shape[0])]
    loss = model.train_on_batch(x=training.__getitem__(index)[0],
            y=[p[idx], training.__getitem__(index)[0]])
    index = index + 1 if (index + 1) * batch_size <= y_pred.shape[0] else 0


with open("python_clusters_with_training.txt", 'w') as out:
    for i in y_pred:
        out.write(str(i) + '\n')


b1 = training.next()[0]
q=5
for i in np.asarray(b1):
    img = np.uint8(np.multiply(np.asarray(i), 255))
    img = img.reshape(256, 256)
    Image.fromarray(np.asarray(img)).save("test_" + str(q) + ".png")
    q += 1

q=5
for i in model.predict_on_batch(b1)[1]:
    print(i.shape)
    img = np.uint8(np.multiply(np.asarray(i), 255))
    img = img.reshape(256, 256)
    Image.fromarray(img).save("pred_" + str(q) + ".png")
    q+=1


