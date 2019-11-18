import cv2
import numpy as np
import time

font = cv2.FONT_HERSHEY_PLAIN
camera = cv2.VideoCapture(0)
# dnn module (Deep Neural Network) is not used for training.
# it is only used for running inference on image or video.
# We use pre-trained yolo weights.


# This line reads weights and config file and creates Neural Network
neuralNet = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Adding object names from coco.names to classes list
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Retrieve layers names (input layer - initial data for neural network, hidden layers - layers between
# input and output layer where all the computations are done, output layer -> produces result of given input)
layer_names = neuralNet.getLayerNames()
# print(layer_names)

# From all layers we choose only the OUTPUT layers that we need from YOLO
output_layers = [layer_names[i[0] - 1] for i in neuralNet.getUnconnectedOutLayers()]
# print(output_layers)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Detecting objects
# Blob is standard array and unified memory interface for the framework.

# Blob - binary large object
# his is a datatype that enables to store large amount of binary data as one object.
# blobFrom image returns returns a blob which is our input image after mean subtraction,
# normalizing and channel swapping

# Blob is used to extract object from image.
starting_time = time.time()
frame_id = 0

while True:

    _, frame = camera.read()
    frame_id += 1
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), mean=(0, 0, 0), swapRB=True, crop=False)

    # for b in blob:
    #     for n, img_blob in enumerate(b):
    #         cv2.imshow(str(n), img_blob)

    # Setting new input value for network (we are passing blobs that we created)
    neuralNet.setInput(blob)
    # Perform a forward pass of the YOLO object detector and gives boxes and assosciated probabilities (confidences)
    outs = neuralNet.forward(output_layers)
    # print(outs)

    # Showing information on the screen

    # Boxes contains the coordinates of rectangle surrounding the object detected
    # Confidences contains confidences about the detection of object from 0 to 1
    boxes = []
    confidences = []
    class_ids = []

    # Loop over each of the layer outputs
    # outs is a list of lists. Each list is a detected object.
    for out in outs:
        # Loop over each of the detections in output
        for detectedObject in out:
            # extract the class id and confidence of the current object detection
            scores = detectedObject[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # object detected!
                # detection[0:4] returns x,y center coordinates of detection and its width and height
                # we need to scale the bounding box back to the size of original image!
                center_x = int(detectedObject[0] * width)
                center_y = int(detectedObject[1] * height)
                w = int(detectedObject[2] * width)
                h = int(detectedObject[3] * height)

                # Rectangle coordinates - top left corner
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # we need to only take indexes, there may be same object detected twice
    # Non maximum suppression takes boxes with highest confidence score and remove multiple other objects
    # from one object
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    # print(indexes)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            color = colors[class_ids[i]]
            # print(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 30), font, 3, (255, 255, 255), 1)
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    # 27 is ESC key
    if key == 27:
        break

# show image wait for any key and close windows.
camera.release()
cv2.destroyAllWindows()
