#feature detection and matching

#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


# Load the images
image1 = cv2.imread('dataset/Image_1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('dataset/Image_2.jpeg', cv2.IMREAD_GRAYSCALE)


# In[3]:


# Initialize the feature detector (SIFT) and feature matcher (Brute-Force)
sift = cv2.SIFT_create()
bf = cv2.BFMatcher()


# In[4]:


# Detect and compute the keypoints and descriptors for the images
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)


# In[5]:


# Match the descriptors using the Brute-Force matcher
matches = bf.match(descriptors1, descriptors2)


# In[6]:


# Sort the matches by distance
matches = sorted(matches, key=lambda x: x.distance)


# In[7]:


# Draw the top N matches
N = 10
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:N], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


# In[8]:


# Display the matched image
cv2.imshow('Feature Matching', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




