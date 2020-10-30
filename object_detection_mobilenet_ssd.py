import cv2
import numpy as np

# Load the pre-trained MobileNet single shot detection (SSD) model
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt',
                               'MobileNetSSD_deploy.caffemodel')

categories = {0: 'background', 1: 'aeroplane',
              2: 'bicycle', 3: 'bird', 4: 'boat',
              5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
              9: 'chair', 10: 'cow', 11: 'diningtable',
              12: 'dog', 13: 'horse', 14: 'motorbike',
              15: 'person', 16: 'pottedplant',
              17: 'sheep', 18: 'sofa', 19: 'train',
              20: 'tvmonitor'}

# defined in list also
classes = ["background", "aeroplane", "bicycle", "bird",
           "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse",
           "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# read image name to check different results
image = cv2.imread('image_3.jpeg')

# create numpy array with same shape with input image
(h, w) = image.shape[:2]
# print(str(h) + str(w))
# print(image.shape)

image_resized = cv2.resize(image, (300, 300))

# create a 4 dimensional blob from resized image and normalizes it
blob = cv2.dnn.blobFromImage(
    image_resized, scalefactor=0.007843, size=(300, 300), mean=127.5)

# feed the blob in to the deep neural network for object detection
net.setInput(blob)
detections = net.forward()
# print(type(detections))

# set random colors for the detected object rectangles
colors = np.random.uniform(low=255, high=0, size=(len(categories), 3))


for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    # print(type(confidence))

    # discard detections with confidence less than 30%
    if confidence > 0.3:
        idx = int(detections[0, 0, i, 1])

        # locate the position of detected object in an image
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # label = "{}: {:.2f}%".format(categories[idx], \
        # confidence * 100)
        label = "{}: {:.2f}%".format(classes[idx],
                                     confidence*100)

        # create a rectangular box around the object
        cv2.rectangle(image, (startX, startY), (endX, endY), colors[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        # along with rectangular box, we will use cv2.putText
        # to write label of the detected object
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, colors[idx], 2)

cv2.imshow("detection", image)
key = cv2.waitKey(0)
if key == ord('s'):
    cv2.imwrite("detected_scene.jpg", image)
    print("Image saved")

cv2.destroyAllWindows()
