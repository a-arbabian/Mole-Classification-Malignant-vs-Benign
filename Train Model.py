#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
import pickle 
import os 

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
resnet_weights_path = os.getcwd() + '/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

X_train = pickle.load(open("X_train.pickle", "rb"))
y_train = pickle.load(open("y_train.pickle", "rb"))
X_val = pickle.load(open("X_val.pickle", "rb"))
y_val = pickle.load(open("y_val.pickle", "rb"))

X_train = X_train/255.0
X_val = X_val/255.0

input_shape = (224,224,3)
lr = 0.001
epochs = 150
batch_size = 64
layer_size = 64
# model = Sequential()
# model.add(ResNet50(include_top=False,
#                  weights= resnet_weights_path,
#                  input_shape=input_shape,
#                  pooling='avg',
#                  classes=2))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(Dense(1, activation='sigmoid'))
# model.layers[0].trainable = False
model = Sequential()
model.add(Conv2D(layer_size, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(layer_size, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(layer_size, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()





# In[2]:


model.compile(optimizer = 'adam' ,
              loss = "binary_crossentropy", 
              metrics=["accuracy"])

#Learning rate decay with ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            mode = 'max',
                                            patience=3, 
                                            verbose=1,
                                            factor=0.7, 
                                            min_lr=1e-7)


# Train model
model.fit(X_train, y_train, validation_data=(X_val,y_val),
                epochs= epochs, batch_size= batch_size, verbose=2,
             callbacks=[learning_rate_reduction])


# In[ ]:


test_loss, test_acc = model.evaluate(X_val, y_val)

print('Test accuracy:', test_acc)
print('Test loss:', test_loss)


# In[ ]:


import time 
# save model
# serialize model to JSON
t = time.time()
model_json = model.to_json()

with open(f"model-valAcc-{test_acc}-{t}.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights(f"model-valAcc-{test_acc}-{t}.h5")
print("Saved model to disk")


# In[ ]:




