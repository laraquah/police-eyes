import streamlit as st
import cv2
import numpy as np
from openvino.runtime import Core
from scipy.spatial.distance import cosine
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from camera_input_live import camera_input_live  # Assuming this is the correct import
import PIL

# Initialize OpenVINO's Inference Engine
ie = Core()

# Load the face detection and face re-identification models
face_model_path = "models2/face-detection-adas-0001.xml"
embedding_model_path = "models2/face-reidentification-retail-0095.xml"

# Load models into OpenVINO
face_net = ie.read_model(model=face_model_path)
face_exec_net = ie.compile_model(model=face_net, device_name="CPU")
embedding_net = ie.read_model(model=embedding_model_path)
embedding_exec_net = ie.compile_model(model=embedding_net, device_name="CPU")

# Input layers for the models
face_input_layer_name = face_net.inputs[0].get_any_name()
embedding_input_layer_name = embedding_net.inputs[0].get_any_name()

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Function to extract embeddings from the face image
def get_face_embedding(image, exec_net, input_layer_name):
    # Resize to input size required by the model
    input_image = cv2.resize(image, (128, 128))  # Adjust size as per model requirements
    input_image = input_image.transpose((2, 0, 1))  # Convert HWC to CHW
    input_image = np.expand_dims(input_image, axis=0)

    # Perform inference to get face embeddings
    infer_request = exec_net.create_infer_request()
    infer_request.infer(inputs={input_layer_name: input_image})
    
    # Get output embeddings
    embedding = infer_request.get_output_tensor().data
    return embedding.flatten()

# Streamlit page configuration
st.title("Police Eyes :cop:")
st.text("Using OpenVINO and Streamlit")

# Upload the reference image
uploaded_file = st.file_uploader("Upload a picture of the person to compare", type=["jpg", "jpeg", "png"])
reference_embedding = None  # Initialize reference embedding

if uploaded_file is not None:
    # Load the reference image
    reference_image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    reference_image = cv2.imdecode(reference_image, 1)
    st.image(reference_image, caption="Uploaded Reference Image", use_column_width=True)

    # Perform face detection and embedding extraction on the reference image
    h, w = reference_image.shape[:2]
    face_input = cv2.resize(reference_image, (672, 384))  # Resize to model input size
    face_input = face_input.transpose((2, 0, 1))  # Convert HWC to CHW
    face_input = np.expand_dims(face_input, axis=0)
    
    # Detect face in reference image
    face_infer_request = face_exec_net.create_infer_request()
    face_infer_request.infer(inputs={face_input_layer_name: face_input})
    detections = face_infer_request.get_output_tensor().data
    
    # Extract the face and get the embedding
    for detection in detections[0][0]:
        confidence = detection[2]
        if confidence > 0.5:
            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)
            face_crop = reference_image[ymin:ymax, xmin:xmax]
            reference_embedding = get_face_embedding(face_crop, embedding_exec_net, embedding_input_layer_name)

# Function to play live camera
def play_live_camera():
    image = camera_input_live()
    uploaded_image = PIL.Image.open(image)
    uploaded_image_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)

    # Here you might want to perform some processing on the uploaded_image_cv
    # For example, running inference on the live image
    # If no processing is needed, you can display it directly

    st.image(uploaded_image_cv, channels="BGR")  # Display live camera feed

# Start Webcam Stream with camera_input_live
st.subheader("Live Camera Input")
play_live_camera()  # Call the function to display live camera feed

if uploaded_file is not None and reference_embedding is not None:
    webrtc_streamer(key="face_comparison", video_transformer_factory=FaceComparison)
else:
    st.warning("Please upload a reference image to start comparison.")

