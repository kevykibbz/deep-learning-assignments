import cv2
import numpy as np

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Read the first frame
ret, previous_frame = video_capture.read()
previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

# Set parameters for the Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate the optical flow
    optical_flow = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, None, None, **lk_params)

    # Select good points for motion estimation
    good_old = optical_flow[0].reshape(-1, 2)
    good_new = optical_flow[1].reshape(-1, 2)

    # Draw the motion vectors
    for (new, old) in zip(good_new, good_old):
        x1, y1 = new.ravel()
        x0, y0 = old.ravel()
        cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow('Motion Estimation', frame)

    # Set the current frame as the previous frame for the next iteration
    previous_gray = current_gray.copy()

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
