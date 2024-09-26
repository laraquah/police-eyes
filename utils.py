import cv2
import numpy as np
from openvino.runtime import Core

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

def predict_image(image, exec_net, input_layer_name, conf_threshold):
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
