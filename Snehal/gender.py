import cv2
import numpy as np

# Load the models
faceProto = "e:/icey/mycode/face detection/opencv_face_detector.pbtxt"
faceModel = "e:/icey/mycode/face detection/opencv_face_detector_uint8.pb"
ageProto = "e:/icey/mycode/face detection/age_deploy.prototxt"
ageModel = "e:/icey/mycode/face detection/age_net.caffemodel"
genderProto = "e:/icey/mycode/face detection/gender_deploy.prototxt"
genderModel = "e:/icey/mycode/face detection/gender_net.caffemodel"

# Load networks
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Define model mean values, age ranges, and gender list
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Function to detect faces and predict gender and age
def facebox(faceNet, frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    
    bbox = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bbox.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return bbox

# Initialize video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture video frame")
        break
    
    # Detect faces in the frame
    faceBoxes = facebox(faceNet, frame)
    
    # Loop over each detected face
    for bbox in faceBoxes:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = f"{gender}, {age}"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Age-Gender Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close windows
video.release()
cv2.destroyAllWindows()
