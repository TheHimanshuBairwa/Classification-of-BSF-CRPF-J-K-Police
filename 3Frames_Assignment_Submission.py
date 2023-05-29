# -*- coding: utf-8 -*-
"""
Created on Mon May 29 16:33:36 2023

@author: EEE
"""




#%%

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#%%
# path to your dataset directory
dataset_dir = 'D:/3Frames Software Labs'

#  image dimensions
img_width, img_height = 150, 150

# number of classes (forces)
num_classes = 3

# batch size for training
batch_size = 32

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Model creation
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training
epochs = 30
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the trained model
model.save('soldier_uniform_classifier.h5')


#%%
# Prediction
def predict_soldier_uniform(image_path):
    loaded_model = tf.keras.models.load_model('soldier_uniform_classifier.h5')
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    normalized_img_array = img_array / 255.0
    prediction = loaded_model.predict(normalized_img_array)
    predicted_class = np.argmax(prediction)
    
    # Mapping predicted class index to force names
    force_names = ['CRPF', 'BSF', 'Jammu & Kashmir Police']
    predicted_force = force_names[predicted_class]
    
    return predicted_force


test_image_path = 'D:/Himanshu/2.jpg'
predicted_force = predict_soldier_uniform(test_image_path)
print("Predicted force:", predicted_force)
