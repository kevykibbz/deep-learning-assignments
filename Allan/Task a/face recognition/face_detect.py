#face_detect.py

#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pathlib
import cv2


# In[3]:


cascade_path=pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"


# In[4]:


print(cascade_path)


# In[5]:


clf=cv2.CascadeClassifier(str(cascade_path))


# In[6]:


# Initialize the video capture
# Use 0 for the default webcam
camera = cv2.VideoCapture(0)  
#camera = cv2.VideoCapture("video.mp4") 


# In[ ]:


while True:
    # Read a frame from the video capture
    ret, frame = camera.read()
    # Check if the frame is not empty
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,width,height) in faces:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255,255,0), 2)
    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




