#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import tensorflow as tf


# In[2]:


# Load the pre-trained MobileNet model
model = tf.keras.applications.MobileNetV2(weights='imagenet')


# In[3]:


# Load the image
image = cv2.imread('dataset/object.jpg')

# Preprocess the image
resized_image = cv2.resize(image, (224, 224))
print(resized_image)
input_image = tf.keras.applications.mobilenet.preprocess_input(resized_image)


# In[4]:


# Add batch dimension
input_image = np.expand_dims(input_image, axis=0)

# Perform object recognition
predictions = model.predict(input_image)


# In[5]:


# Get the top 5 predicted classes and their probabilities
top_predictions = tf.keras.applications.mobilenet.decode_predictions(predictions, top=5)[0]


# In[ ]:


# Display the top predictions
for pred in top_predictions:
    class_name = pred[1]
    probability = pred[2]
    print(f"Class: {class_name}, Probability: {probability}")

# Display the image
cv2.imshow('Object Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




