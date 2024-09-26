import cv2
import numpy as np
from openvino.runtime import Core

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


def load_model(model_path):
    """Load and compile the OpenVINO model."""
    ie = Core()
    model = ie.read_model(model=model_path)
    exec_model = ie.compile_model(model=model, device_name="CPU")
    return exec_model

def preprocess_image(image, size=(672, 384)):
    """Preprocess the input image for model inference."""
    image_resized = cv2.resize(image, size)
    image_transposed = image_resized.transpose((2, 0, 1))  # Convert HWC to CHW
    return np.expand_dims(image_transposed, axis=0)  # Add batch dimension

def postprocess_detections(detections, confidence_threshold):
    """Filter detections based on confidence threshold."""
    valid_detections = []
    for detection in detections[0][0]:  # Adjust based on your output structure
        confidence = detection[2]
        if confidence > confidence_threshold:
            valid_detections.append(detection)
    return valid_detections

def predict_image(image, exec_net, input_layer_name):
    """Run inference on the image and visualize results."""
    input_image = preprocess_image(image)
    
    # Perform inference
    infer_request = exec_net.create_infer_request()
    infer_request.infer(inputs={input_layer_name: input_image})
    detections = infer_request.get_output_tensor().data

    # Process detections and visualize
    for detection in postprocess_detections(detections, conf_threshold):
        class_id, confidence, xmin, ymin, xmax, ymax = detection
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Draw green box
        cv2.putText(image, f'Class: {class_id}, Conf: {confidence:.2f}', 
                    (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image
