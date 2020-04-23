# CVND-Facial-Keypoints-Detection

This project is about defining and training a CNN to perform facial keypoint detection, and using computer vision techniques to transform images of faces. Facial keypoints (also called facial landmarks) mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, etc.

The project pipeline starts with detecting all the faces in an image using a face detector, OpenCV's pre-trained Haar Cascade classifier was used.

Haar Cascade classifier: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html

The next step is to pre-process that image and feed it into a CNN facial keypoint detector, I used PyTorch framework to create the model, the CNN consisted of 4 Convolutional layers with batch normalization and dropout layers to avoid overfitting, followed by 3 fully connected layers. This facial keypoints dataset consists of 5770 color images extracted from the YouTube Faces Dataset. The model was trained for 5 epochs reaching an average loss of = 0.072.

The model output is 68 keypoints, with coordinates (x, y), for that face, which I used later to add filters on the detected faces in an input live stream from a camera, running only on the CPU not the GPU without any delay in the feed.


#### USE KEYBOARD BUTTONS TO CHANGE FILTERS IN LivePrediction.py SCRIPT
###### 1 -> NORMAL CAMERA
###### 2 -> DETECTED FACE
###### 3 -> DETECTED KEYPOINTS
###### 4 -> FILTER 1
###### 5 -> FILTER 2
###### q -> QUIT

