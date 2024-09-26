import streamlit as st
import cv2
import numpy as np
from openvino.runtime import Core
from camera_input_live import camera_input_live  # Assuming this is the correct import
import PIL
import utils

st.title("Police Eyes :cop:")
st.text("Using OpenVINO and Streamlit")

# Radio button to choose source
source_radio = st.radio("Choose your input source:", ("UPLOAD", "WEBCAM"))

# Upload the reference image if source is UPLOAD
reference_embedding = None  # Initialize reference embedding

if source_radio == "UPLOAD":
    uploaded_file = st.file_uploader("Upload a picture of the person to compare", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the reference image
        reference_image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        reference_image = cv2.imdecode(reference_image, 1)
        st.image(reference_image, caption="Uploaded Reference Image", use_column_width=True)
 
def play_live_camera():
    image = camera_input_live()
    uploaded_image = PIL.Image.open(image)
    uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
    visualized_image = utils.predict_image(uploaded_image_cv)
    st.image(visualized_image, channels = "BGR")

if source_radio == "WEBCAM":
    play_live_camera()
