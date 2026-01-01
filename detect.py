import cv2
import numpy as np

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    "cascades/haarcascade_frontalface_default.xml"
)

# Load Age & Gender models
age_net = cv2.dnn.readNet(
    "models/age_net.caffemodel",
    "models/age_deploy.prototxt"
)

gender_net = cv2.dnn.readNet(
    "models/gender_net.caffemodel",
    "models/gender_deploy.prototxt"
)

# Labels
AGE_BUCKETS = [
    "0-2", "4-6", "8-12", "15-20",
    "25-32", "38-43", "48-53", "60-100"
]

GENDER_LIST = ["Male", "Female"]

# Mean values (important)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Read image
image = cv2.imread("kid.jpg")
image = cv2.resize(image, (800, 800))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(50, 50)
)
for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w]

    if face.size == 0:
        continue

    # Create blob
    blob = cv2.dnn.blobFromImage(
        face,
        1.0,
        (227, 227),
        MODEL_MEAN_VALUES,
        swapRB=False
    )

    # Gender prediction
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDER_LIST[gender_preds[0].argmax()]

    # Age prediction
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]

    label = f"{gender}, {age}"

    # Draw box and label
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(
        image,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

# Show result
cv2.imshow("Age & Gender Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
