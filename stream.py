# ______________________________________________
# Imports for the camera stream module         |
# ____________________________________________|
import cv2
import os
from dotenv import load_dotenv
# ________________________________________________
# Load video stream from Wisenet without the     |
# need for web browser                          |
# _____________________________________________|

# Load enviornment variables from .env file
load_dotenv()
rtsp_url=os.getenv("RTSP_URL")
if not rtsp_url:
    print("Error: RTSP_URL not found in .env file")
    exit()

# Connect to the video stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the connection was successful
if not cap.isOpened():
    print("Error: Could not open video stream. Check the RTSP URL and credentials.")
    exit()

# Read and display video frames in a loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check connection or credentials.")
        break
        
    # Zoom out by resizing the frame to 50% of its original size
    height, width = frame.shape[:2]
    zoomed_out_frame = cv2.resize(frame, (int(width * 0.5), int(height * 0.5)))
    
    # Display the resized video frame
    cv2.imshow("Wisenet Camera Stream", zoomed_out_frame)
    
    # Press 'q' to quit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()