#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:18:47 2022

@author: Stephen L
"""
#Command line arguments
# python Part_B_cmdline.py True 64 0 128 0 Exception

import gdown
import wget
import sys
import pathlib
import numpy as np
from PIL import Image
import tensorflow.keras as tfk
import os
import glob
import wandb
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt
import math
import requests
import zipfile

augment_data = sys.argv[1]
batch_size =sys.argv[2]
dropout = sys.argv[3]
dense_layer = sys.argv[4]
layer_freeze = sys.argv[5]
pre_model = sys.argv[6]

print('Passed arguments are:')
print('augment_data:',augment_data)
print('batch_size:',batch_size)
print('dropout:',dropout)
print('dense_layer:',dense_layer)
print('layer_freeze:',layer_freeze)
print('pre_model:',pre_model)



#Data_preperation

def datagen(batch_size, augment_data):

   # path = os.getcwd()
   # # url ='https://www.mca.gov.in/Ministry/pdf/FAQ_pdf_size.pdf'
   # url ='https://storage.googleapis.com/wandb_datasets/nature_12K.zip'
   # path1= path+'/inaturalist_12K'
   # gdown.download(url, path1, quiet=False)
   
   # j= (path + '/inaturalist_12K')
   # with zipfile.ZipFile(j,"r") as zip_ref:
   #   zip_ref.extractall(path) 
    
   Data_dir=pathlib.Path('inaturalist_12K') # Set path to the right directory";
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
       shuffle=True,
       seed=42
       )
   test_data = test_rawdata.flow_from_directory(
       test_path, 
       target_size=(256, 256), 
       batch_size=batch_size,
       color_mode="rgb",
       class_mode='categorical',
       shuffle=True,
       seed=42
       )
   return  train_data, val_data, test_data


def define_model(model_name, activation_function_dense, dense_layer, dropout,image_size, pre_layer_train=None):
    
    #input_image_shape=(256,256,3)
    input_shape=(image_size, image_size, 3)
    input_ = tfk.Input(shape = input_shape)

    # add a pretrained model without the top dense layer
    if model_name == 'ResNet50':
      pretrained_model = tfk.applications.ResNet50(include_top = False, weights='imagenet',input_tensor = input_)
    elif model_name == 'InceptionV3':
      pretrained_model = tfk.applications.InceptionV3(include_top = False, weights='imagenet',input_tensor = input_)
    elif model_name == 'InceptionResNetV2':
      pretrained_model = tfk.applications.InceptionResNetV2(include_top = False, weights='imagenet',input_tensor = input_)
    else:
      pretrained_model = tfk.applications.Xception(include_top = False, weights='imagenet',input_tensor = input_)

    for layer in pretrained_model.layers:
      layer.trainable=False
    
    model = tfk.models.Sequential()
    model.add(pretrained_model)#add pretrained model
    model.add(Flatten()) # The flatten layer is essential to convert the feature map into a column vector
    model.add(Dense(dense_layer, activation=activation_function_dense))#add a dense layer
    model.add(Dropout(dropout)) # For dropout
    model.add(Dense(10, activation="softmax"))#softmax layer

    return model



train_data, val_data, test_data =datagen(batch_size, augment_data)
print("Data Recieved")

activation_func_dense = "relu"
image_size = 256

model=define_model(pre_model, activation_func_dense, dense_layer, dropout,image_size)
print(model.count_params())
print("Training done")

TRAIN_STEP_SIZE = int(train_data.n)//int(train_data.batch_size)
print(TRAIN_STEP_SIZE)
VAL_STEP_SIZE = int(val_data.n)//int(val_data.batch_size)
print(VAL_STEP_SIZE)

model.compile(optimizer=Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Early Stopping 
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

model_det = model.fit(train_data,
      steps_per_epoch = TRAIN_STEP_SIZE,
      validation_data = val_data,
      validation_steps = VAL_STEP_SIZE,
      epochs=10, 
      # callbacks=[WandbCallback(data_type="image", generator=val_data), earlyStopping, best_val_check],
      #verbose=2)

#For fine tuning, unfreeze certain layers
if layer_freeze:  
    fine_tune=math.floor((layer_freeze/100.0)*len(model.layers))
    for layer in model.layers[-fine_tune:]:
        layer.trainable=True
  

print("Fine tuning")
model.compile(optimizer=Adam(learning_rate=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model_det_fine = model.fit(train_data,
      steps_per_epoch = TRAIN_STEP_SIZE,
      validation_data = val_data,
      validation_steps = VAL_STEP_SIZE,
      epochs=10, 
      callbacks=[WandbCallback(data_type="image", generator=val_data), earlyStopping, best_val_check],
      verbose=2)
test_loss, test_acc = model.evaluate(test_data)
print('Test accuracy :', test_acc)