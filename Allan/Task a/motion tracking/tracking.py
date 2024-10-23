#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam


# In[3]:


# Initialize the background subtractor
background_subtractor = cv2.createBackgroundSubtractorMOG2()


# In[ ]:


while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()
    
    # Apply background subtraction
    mask = background_subtractor.apply(frame)
    
    # Perform morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding rectangles around moving objects
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust the area threshold as needed
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Motion Tracking', frame)
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




