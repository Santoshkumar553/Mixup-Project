import cv2
import numpy as np

# Load pre-trained model and configuration file
model_path = "MobileNetSSD_deploy.caffemodel"
config_path = "MobileNetSSD_deploy.prototxt"

# Load the model
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# List of class labels for MobileNet-SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to improve processing speed
    h, w = frame.shape[:2]
    resized_frame = cv2.resize(frame, (300, 300))

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(resized_frame, 0.007843, (300, 300), 127.5)

    # Perform forward pass to get detections
    net.setInput(blob)
    detections = net.forward()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Filter detections by confidence threshold
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Obstacle Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
