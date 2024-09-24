import streamlit as st
import cv2
import numpy as np
from openvino.runtime import Core
from scipy.spatial.distance import cosine

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
if uploaded_file is not None:
    # Load the reference image
    reference_image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    reference_image = cv2.imdecode(reference_image, 1)
    st.image(reference_image, caption="Uploaded Reference Image", use_column_width=True)

    # Perform face detection and embedding extraction on the reference image
    reference_embedding = None
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

# Start Webcam Stream
run = st.checkbox('Run Webcam and Compare')
FRAME_WINDOW = st.image([])


reference_embedding = None
if run and reference_embedding is not None:
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video", icon="🚨")
            break

        # Resize frame for face detection
        h, w = frame.shape[:2]
        face_input = cv2.resize(frame, (672, 384))
        face_input = face_input.transpose((2, 0, 1))
        face_input = np.expand_dims(face_input, axis=0)

        # Perform face detection
        face_infer_request.infer(inputs={face_input_layer_name: face_input})
        detections = face_infer_request.get_output_tensor().data

        for detection in detections[0][0]:
            confidence = detection[2]
            if confidence > 0.5:
                xmin = int(detection[3] * w)
                ymin = int(detection[4] * h)
                xmax = int(detection[5] * w)
                ymax = int(detection[6] * h)

                # Draw rectangle around face
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                # Extract face for embedding generation
                face_crop = frame[ymin:ymax, xmin:xmax]
                current_embedding = get_face_embedding(face_crop, embedding_exec_net, embedding_input_layer_name)

                # Compare the embeddings
                similarity = cosine_similarity(reference_embedding, current_embedding)
                if similarity > 0.5:  # Adjust threshold as needed
                    cv2.putText(frame, "Criminal Identified!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame in Streamlit
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    cv2.destroyAllWindows()
else:
    st.warning("Please upload a reference image to start comparison.")

