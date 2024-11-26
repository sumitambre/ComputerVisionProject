import streamlit as st
import cv2
import tempfile
import time
from ultralytics import YOLO  # Assuming YOLOv8 from ultralytics library
import numpy as np
import torch


print("GPU Available:", torch.cuda.is_available())
print("Device Name:", torch.cuda.get_device_name(0))

# Load the YOLO model
model = YOLO("yolov8s.pt").to("cuda")  # Replace with the appropriate model path

# Set Streamlit Page Configuration
st.set_page_config(page_title="YOLO Object Detection", layout="centered")

# Title and Description
st.title("YOLO Object Detection App")
st.write("Upload a video or use your webcam for real-time object detection using YOLO.")

# Sidebar for Settings
st.sidebar.header("Settings")
detection_classes = st.sidebar.multiselect(
    "Select Objects to Detect:",
    options=["person", "car", "bicycle", "dog", "cat", "truck", "motorbike","handbags"],
    default=["person", "car"]
)

# Video Source Selection
video_source = st.sidebar.selectbox("Select Video Source", ["Upload Video", "Use Webcam"])

# Video Upload Option
uploaded_file = None
if video_source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a Video File", type=["mp4", "avi", "mov"])

# Start/Stop Detection Button
start_detection = st.sidebar.button("Start Detection")

# Main UI for Video Feed Display
stframe = st.empty()  # Placeholder for video frame

def process_frame(frame, selected_classes):
    """
    Process a video frame with YOLO and draw detection results.
    """
    results = model(frame)  # Perform detection
    print(f"result for frame: {results[0].boxes}")
    annotated_frame = frame.copy()
    
    for result in results[0].boxes:
        # Extract detection info
        class_id = int(result.cls)
        confidence = float(result.conf)  # Convert to float
        bbox = result.xyxy.numpy()[0]  # Bounding box coordinates
        
        # Check if class is selected
        class_name = model.names[class_id]
        if class_name in selected_classes and confidence >0.7:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"{class_name} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    
    return annotated_frame

# Detection Logic
if start_detection:
    if video_source == "Use Webcam":
        # Access webcam
        cap = cv2.VideoCapture(0)
    elif uploaded_file is not None:
        # Access uploaded video
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file.name)
    else:
        st.error("Please upload a video file to start detection.")
        cap = None
    
    if cap is not None:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process the frame
            processed_frame = process_frame(frame, detection_classes)
            # st.write(f"selected detection option is {detection_classes} ")
            # Convert processed frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Add delay to simulate real-time processing
            time.sleep(0.03)
        
        cap.release()
    else:
        st.warning("Unable to access video source.")
else:
    st.info("Click 'Start Detection' to begin.")


