import cv2
import numpy as np

# Loading Yolo
net = cv2.dnn.readNet("/home/ren_cosmos/projects/python_projects/object detection/object_detection_0/yolov3.weights",
                      "/home/ren_cosmos/projects/python_projects/object detection/object_detection_0/yolov3.cfg")
classes = []
with open("/home/ren_cosmos/projects/python_projects/object detection/object_detection_0/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading video using Webcam
video = cv2.VideoCapture(0)
classes_detected = []
while True:
    check, frame = video.read()
    frame = cv2.resize(frame, None, fx=2, fy=2)
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(
        frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            # print("class_id = ", class_id)
            confidence = scores[class_id]
            if confidence > 0.5:

                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)
            
            if label not in classes_detected:
                classes_detected.append(label)

            #displaying number of objects detected
            cv2.putText(frame, "detected objects = {}".format(len(classes_detected)), (0, height), font, 3, color, 3)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)

    # Stopping detection with keyboard input
    if key == ord('x'):
        break

cv2.destroyAllWindows()