#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions


# In[2]:


# Load pre-trained VGG16 model
model = tf.keras.applications.VGG16(weights='imagenet')


# In[3]:


# Define directories
base_dir = 'dataset'
incoming_data_dir = os.path.join(base_dir, 'incoming_data')
mixed_data_dir = os.path.join(base_dir, 'mixed_data')


# In[5]:


if not os.path.exists(mixed_data_dir):
    
    # Create mixed_data directory if it doesn't exist
    os.makedirs(mixed_data_dir, exist_ok=True)
    
    # Iterate through cat and dog directories and mix images
    for animal_dir in ['cat', 'dog']:
        animal_path = os.path.join(incoming_data_dir, animal_dir)
        for filename in os.listdir(animal_path):
            img_path = os.path.join(animal_path, filename)
            img = load_img(img_path, target_size=(224, 224))  # Load image
            img_array = img_to_array(img)  # Convert image to array
            img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
            img_array = preprocess_input(img_array)  # Preprocess input
            mixed_img_path = os.path.join(mixed_data_dir, filename)
            np.save(mixed_img_path, img_array)  # Save mixed image


# In[6]:


# Make predictions on mixed images
for filename in os.listdir(mixed_data_dir):
    img_array = np.load(os.path.join(mixed_data_dir, filename))
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=1)[0]  # Get top prediction
    label = decoded_preds[0][1]  # Get label
    print(f"Image: {filename}, Predicted class: {label}")


# In[ ]:




