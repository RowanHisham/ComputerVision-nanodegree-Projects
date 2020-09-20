import numpy as np
import os
import cv2
import torch
from models import *
from enum import Enum

def detectFace(image):
    """
    Detect faces using Haar Cascade Classifier and return (x, y, w, h) for each detected face
    """
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.2, 10,30)
    return faces


def drawFaces(finalImage, faces):
    """
        Draw rectangle around each detected face
    """
    for (x, y, w, h) in faces:
        cv2.putText(finalImage, 'Face Detected', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.rectangle(finalImage, (x, y), (x + w, y + h), (255, 0, 0), 3)


def detectKeyPoints(image, faces):
    """
        Predict facial key-points using the loaded weights of the trained model
    """
    padding = 50
    # loop over the detected faces from haar cascade
    for (x, y, w, h) in faces:
        # Select the region of interest that is the face in the image
        roi = image[y - padding:y + h + padding, x - padding:x + w + padding]
        originalSize = roi.shape
        if(roi.shape[0] != roi.shape[1]):
            return None

        roi = cv2.resize(roi, (224, 224))
        grayscale = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        grayscale = grayscale / 255.0
        img = grayscale.reshape(1, grayscale.shape[0], grayscale.shape[1])
        img = torch.from_numpy(img).float()
        img = torch.unsqueeze(img, 0)

        output_pts = net(img)
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        predicted_key_pts = output_pts.data.numpy()
        predicted_key_pts = predicted_key_pts * 50.0 + 100
        #retrun first detected face
        return predicted_key_pts , roi , originalSize,(x - padding,y - padding,w + padding*2,h + padding*2)
    return None


def drawKeyPoints(finalImage, output):
    """
        Draw a circle on each detected key-point on the face
    """
    predicted_key_pts, roi, originalSize, (x, y, w, h) = output
    for key in predicted_key_pts[0]:
        roi = cv2.circle(roi, tuple(key), 1, (0, 0, 255), 1)
    roi = cv2.resize(roi, (originalSize[1], originalSize[0]))
    finalImage[y:y + h, x:x + w] = roi



class Buttons(Enum):
    NORMAL_CAMERA = 0
    DETECT_FACE = 1
    DETECT_KEYPOINTS = 2
    ADD_FILTER1 = 3
    ADD_FILTER2 = 4
    QUIT = 6

def checkKeyPressed():
    """
        USE KEYBOARD BUTTONS TO CHANGE FILTERS
            1 -> NORMAL CAMERA
            2 -> DETECTED FACE
            3 -> DETECTED KEY-POINTS
            4 -> FILTER 1
            5 -> FILTER 2
            q -> QUIT
    """
    keyPressed = cv2.waitKey(1) & 0xFF
    if keyPressed == ord('q'):
        return Buttons.QUIT
    elif keyPressed == ord('1'):
        return Buttons.NORMAL_CAMERA
    elif keyPressed == ord('2'):
        return Buttons.DETECT_FACE
    elif keyPressed == ord('3'):
        return Buttons.DETECT_KEYPOINTS
    elif keyPressed == ord('4'):
        return Buttons.ADD_FILTER1
    elif keyPressed == ord('5'):
        return Buttons.ADD_FILTER2
    return None

def addFilter(filter,finalImage, output, x,y,w,h):
    """
        Add filters on top of the detected face using predicted facial key-points
    """
    _, roi, originalSize, (x2, y2, w2, h2) = output
    # resize filter
    filter = cv2.resize(filter, (w, h), interpolation=cv2.INTER_CUBIC)
    # get region of interest on the face to change
    roi_color = roi[y:y + h, x:x + w]
    # find all non-transparent pts
    ind = np.argwhere(filter[:, :, 3] > 0)
    # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
    if(ind.shape[0] > roi_color.shape[0]*roi_color.shape[1]):
        return
    for i in range(3):
        roi_color[ind[:, 0], ind[:, 1], i] = filter[ind[:, 0], ind[:, 1], i]
    # set the area of the image to the changed region with sunglasses
    roi[y:y + h, x:x + w] = roi_color
    roi = cv2.resize(roi, (originalSize[1], originalSize[0]))
    finalImage[y2:y2 + h2, x2:x2 + w2] = roi


def displayImage():
    """
        Add how-to-use text on the image and display it
    """
    cv2.putText(finalImage,
                'Press Buttons 1:Regular Camera     2:Detected Face    3:Detected Keypoints    4:Filter1     5:Filter2   q:Quit',
                (10, 480 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Frame", finalImage)


"""
    Load Model
"""
package_dir = os.path.dirname(os.path.abspath(__file__))
net = Net()
net.load_state_dict(torch.load(package_dir + '/saved_models/model4.pt'))
net.eval()


"""
    Load Filters
"""
filter1 = cv2.imread(package_dir + '/images/filter1.png',-1)
filter2 = cv2.imread(package_dir + '/images/filter2.png',-1)
previousKey = None
cap = cv2.VideoCapture(0)


"""
    MAIN LOOP
"""
while(True):
    _ , frame = cap.read()
    frame = cv2.resize(frame, (720, 480))
    finalImage = frame.copy()

    buttonPressed = checkKeyPressed()
    if buttonPressed is None:
        buttonPressed = previousKey
    else:
        previousKey = buttonPressed

    if buttonPressed == Buttons.QUIT:
        break

    if buttonPressed is not None and buttonPressed is not Buttons.NORMAL_CAMERA:
        faces = detectFace(frame)
        if faces is not None:
            if buttonPressed == Buttons.DETECT_FACE:
                drawFaces(finalImage, faces)
            else:
                output = detectKeyPoints(frame, faces)
                if output is not None:
                    if buttonPressed == Buttons.DETECT_KEYPOINTS:
                        drawKeyPoints(finalImage, output)
                    elif buttonPressed == Buttons.ADD_FILTER1:
                        predicted_key_pts, _, _, _ = output
                        addFilter(filter1,finalImage= finalImage, output=output,
                                                x = int(predicted_key_pts[0][17][0]-10),
                                                y = int(predicted_key_pts[0][17][1]),
                                                h = int(abs(predicted_key_pts[0][25][1] - predicted_key_pts[0][29][1])),
                                                w = int(abs(predicted_key_pts[0][16][0] - predicted_key_pts[0][2][0])))
                    elif buttonPressed == Buttons.ADD_FILTER2:
                        predicted_key_pts, _, _, _ = output
                        addFilter(filter2, finalImage= finalImage, output=output,
                                                x=int(predicted_key_pts[0][17][0] - 20),
                                                y=int(predicted_key_pts[0][17][1] - 70),
                                                h=int(abs(predicted_key_pts[0][20][1] - predicted_key_pts[0][9][1]) + 40),
                                                w=int(abs(predicted_key_pts[0][2][0] - predicted_key_pts[0][15][0]) + 30))
    displayImage()