#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 13:47:14 2022

@author: Stephen L
"""
# python Part_A_cmdline.py 64 True 64 True 0.15 128 3

import sys
import pathlib
import numpy as np
from PIL import Image
import os
import glob
import random
from keras.models import load_model
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from collections import defaultdict
import requests

# Constant params
filter_pool_size = [(2, 2), (2, 2), (2, 2), (2, 2), (2, 2)]
image_size = 256
activation_func_convolution = ["relu", "relu", "relu", "relu", "relu"]
activation_func_dense = "relu"


no_of_filters = sys.argv[1]
augment_data = sys.argv[2]
batch_size = sys.argv[3]
batch_normalisation =sys.argv[4]
dropout = sys.argv[5]
dense_layer = sys.argv[6]
filter_convol_size = sys.argv[7]


print('Passed arguments are:')
print('no_of_filters:', no_of_filters)
print('augment_data:',augment_data)
print('batch_size:',batch_size)
print('batch_normalisation:', batch_normalisation)
print('dropout:',dropout)
print('dense_layer:',dense_layer)
print('filter_convol_size:',filter_convol_size)

# Data preperation

def datagen(batch_size, augment_data):
  
  Data_dir=pathlib.Path('inaturalist_12K') 
  # augment_data=False
  train_path = os.path.join(Data_dir, "train")
  test_path = os.path.join(Data_dir, "val")

  if augment_data:
    train_rawdata = ImageDataGenerator(rescale=1./255,
                                      rotation_range=90,
                                      zoom_range=0.2,
                                      shear_range=0.2,
                                      validation_split=0.1,
                                      horizontal_flip=True)
    test_rawdata = ImageDataGenerator(rescale=1./255)

  else:
    train_rawdata = ImageDataGenerator(rescale=1./255, validation_split=0.1)
    test_rawdata = ImageDataGenerator(rescale=1./255)

  train_data = train_rawdata.flow_from_directory(
      train_path, 
      target_size=(256, 256), 
      batch_size=batch_size, 
      subset="training",
      color_mode="rgb",
      class_mode='categorical',
      shuffle=True,
      seed=42
      )
  
  val_data = train_rawdata.flow_from_directory(
      train_path, 
      target_size=(256, 256), 
      batch_size=batch_size, 
      subset="validation",
      color_mode="rgb",
      class_mode='categorical',
      #shuffle=True,
      seed=42
      )
  test_data = test_rawdata.flow_from_directory(
      test_path, 
      target_size=(256, 256), 
      batch_size=batch_size,
      color_mode="rgb",
      class_mode='categorical',
      # shuffle=True,
      seed=42
      )
  
  return  train_data, val_data, test_data


# CNN Model

def CNN_model(activation_func_convolution, activation_func_dense, no_of_filters, filter_convol_size, filter_pool_size, batch_normalisation, dense_layer, dropout,image_size):

  no_of_classes =10
  filter_sizes =[]
  tf.keras.backend.clear_session()

  filter_sizes.append(no_of_filters)
  for layer_num in range(1,3):
      filter_sizes.append(no_of_filters*(2**layer_num))
      filter_sizes.append(no_of_filters*(2**layer_num))
  model = Sequential()
  model.add(Conv2D(filter_sizes[0], kernel_size=(filter_convol_size,filter_convol_size), input_shape=(image_size, image_size, 3), data_format="channels_last"))
  if batch_normalisation:
      model.add(BatchNormalization())
  model.add(Activation(activation_func_convolution[0]))
  model.add(MaxPooling2D(pool_size=filter_pool_size[0] ))

  for i in range(1, 5):
      model.add(Conv2D(filter_sizes[i], kernel_size=(filter_convol_size,filter_convol_size)))
      if batch_normalisation:
          model.add(BatchNormalization())
      model.add(Activation(activation_func_convolution[i]))
      model.add(MaxPooling2D(pool_size=filter_pool_size[i]))
  
  # Converting the feature map into a column vector and regularisation
  model.add(Flatten()) 
  model.add(Dense(dense_layer, activation=activation_func_dense))
  model.add(Dropout(dropout)) 
  model.add(Dense(no_of_classes, activation="softmax")) 

  return model



# Data preperation
train_data, val_data, test_data =datagen(batch_size, augment_data)

# Model creation
model = CNN_model(activation_func_convolution,activation_func_dense , no_of_filters, filter_convol_size, filter_pool_size, batch_normalisation, dense_layer, dropout ,image_size)
print(model.count_params())

model.compile(optimizer=Adam(learning_rate=0.0001), metrics = ['accuracy'], loss = 'categorical_crossentropy')

train_step_size = train_data.n//train_data.batch_size
print(train_step_size)
val_step_size = val_data.n//val_data.batch_size
print(val_step_size)

model_det = model.fit(train_data,
          steps_per_epoch = train_step_size,
          validation_data = val_data,
          validation_steps = val_step_size,
          epochs=1, 
          #callbacks=[WandbCallback(data_type="image", generator=val_data), earlyStopping, mc],
          verbose=2)

# best_model = load_model("best_model_1st.h5")
# input_image_shape = (256, 256, 3)

# Test accuracy

test_predictions = np.argmax(int(best_model.predict(test_data), axis=-1))
test_loss, test_accuracy = best_model.evaluate(test_data)
print(f"Test accuracy = {test_accuracy*100} %")