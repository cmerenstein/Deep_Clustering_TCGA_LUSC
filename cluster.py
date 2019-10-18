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
from keras.engine.topology import Layer, InputSpec
import keras
import keras.backend as K
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import metrics

encoder_file = open("models/encoder_save_10_15_256px_1024_100k.json").read()
Encoder = model_from_json(encoder_file)

autoencoder = model_from_json(open("models/model_save_10_15_256px_100k.json").read())

## on a new set
v_datagen = ImageDataGenerator(rescale = 1./255)
valid = v_datagen.flow_from_directory("sampled_tiles/larger_valid_256_var_filt", class_mode="input",
        batch_size=8, target_size=(256, 256), color_mode = "grayscale",
        shuffle = False, save_to_dir = "transformed/save_transformed_06/", save_prefix = "grey_")

n_clusters = 50

clustering_layer = ClusteringLayer(n_clusters, name="clustering")(Encoder.output)
model = Model(inputs = Encoder.input, outputs = clustering_layer)
model.compile(optimizer=SGD(0.0001, 0.9), loss='kld')

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(Encoder.predict_generator(valid))
y_pred_last = np.copy(y_pred)

#initialize centers 
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

loss = 0
index = 0
maxiter = 100000
update_interval = 200
index_array = np.arange(0, y_pred.shape[0])
batch_size = 8
tol = 0.001

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict_generator(valid)
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
    loss = model.train_on_batch(x=valid.__getitem__(index)[0], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= y_pred.shape[0] else 0

with open("python_clusters.txt", 'w') as out:
    for i in y_pred:
        out.write(str(i) + '\n')

class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
#
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
#
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), 
		initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
#
    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters),
		axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q
#
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters
#
    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

