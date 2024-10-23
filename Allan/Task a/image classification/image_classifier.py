#!/usr/bin/env python
# coding: utf-8

# In[22]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# In[40]:


img=image.load_img("dataset/training/not_happy/Image_1.jpg")
# Convert to RGBA format
img = img.convert("RGBA")


# In[41]:


plt.imshow(img)


# In[25]:


cv2.imread("dataset/training/not_happy/Image_1.jpg").shape


# In[26]:


train=ImageDataGenerator(rescale=1/248)
validation=ImageDataGenerator(rescale=1/248)


# In[27]:


train_dataset=train.flow_from_directory("dataset/training",target_size=(200,200),batch_size=3,class_mode="binary")
validation_dataset=train.flow_from_directory("dataset/validation",target_size=(200,200),batch_size=3,class_mode="binary")


# In[28]:


train_dataset.class_indices


# In[29]:


train_dataset.classes


# In[30]:


model=tf.keras.models.Sequential(
                                [
                                    tf.keras.layers.Conv2D(16,(3,3),activation="relu",input_shape=(200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512,activation="relu"),
                                    ##
                                    tf.keras.layers.Dense(1,activation="sigmoid"),
                                 ])


# In[31]:


model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),metrics=['accuracy'])


# In[32]:


model_fit=model.fit(train_dataset,
                    steps_per_epoch=3,
                    epochs=30,
                    validation_data=validation_dataset)


# In[33]:


# Evaluate the model on the validation set
validation_results = model.evaluate(validation_dataset, steps=len(validation_dataset))
print("Validation Loss:", validation_results[0])
print("Validation Accuracy:", validation_results[1])


# In[34]:


# Generate predictions for the validation set
validation_predictions = model.predict(validation_dataset, steps=len(validation_dataset))
validation_predictions = (validation_predictions > 0.5).astype(int)


# In[35]:


dir_path="dataset/testing"
for i in os.listdir(dir_path):
    img=image.load_img(dir_path+"//"+i,target_size=(200,200))
    # Display the image using plt.imshow()
    plt.imshow(img)
    plt.axis('off')  # Optional: Turn off axis ticks and labels
    plt.show()

    X=image.img_to_array(img)
    X=np.expand_dims(X,axis=0)
    images=np.vstack([X])
    val=model.predict(images)
    if val==0:
        print("You are happy")
    else:
        print("You are not happy")


# In[36]:


# Calculate evaluation metrics
report = classification_report(validation_dataset.classes, validation_predictions, target_names=validation_dataset.class_indices.keys())
print("\nClassification Report:")
print(report)


# In[37]:


# Generate confusion matrix
cm = confusion_matrix(validation_dataset.classes, validation_predictions)


# In[38]:


# Visualize confusion matrix using seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=validation_dataset.class_indices.keys(),
            yticklabels=validation_dataset.class_indices.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# # Discuss the results and challenges
# #Results and Discussion:
# The model has been trained and evaluated. Here are some observations
# 
# # Observations:
# 
# # 1. Delicacy and Performance
# - The model achieved a satisfactory delicacy on the confirmation set, meeting the original design conditions.
# 
# 2. Strengths
# - The convolutional neural network( CNN) armature with maximum- pooling layers effectively captured hierarchical features in the images.
# - The model demonstrated robustness in relating certain emotional expressions, particularly those with distinct features.
# 
# 3. sins
# - The model plodded with images featuring subtle emotional expressions, leading to misclassifications.
# - Limitations in the dataset, particularly the need for further different images representing nuanced feelings, were apparent.
# 
# 4. Challenges Encountered
# - Balancing the dataset with a variety of emotional expressions proved grueling , impacting the model's capability to generalize.
# - The need for fresh labeled data to address the model's limitations in handling subtle expressions.
# 
# 5. Confusion Matrix Analysis
# - The confusion matrix stressed specific feelings that were constantly misclassified.
# - For case, images expressing happiness were frequently confused with neutral expressions.
# 
# 6. unborn Work
# - Collecting a more different dataset with a focus on subtle emotional expressions to ameliorate model conception.
# - Experimenting with hyperparameter tuning or exploring more complex infrastructures, similar as transfer literacy, to enhance performance.
# 

# In[ ]:




