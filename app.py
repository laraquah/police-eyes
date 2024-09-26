import streamlit as st
import cv2
import numpy as np
from openvino.runtime import Core
from scipy.spatial.distance import cosine
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from camera_input_live import camera_input_live  # Assuming this is the correct import
import PIL

def play_live_camera():
    image = camera_input_live()
    uploaded_image = PIL.Image.open(image)
    uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    visualized_image = utils.predict_image(uploaded_image_cv, conf_threshold)
    st.image(visualized_image, channels = "BGR")

if source_radio == "WEBCAM":
    play_live_camera()
