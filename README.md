# Computer Vision Nanodegree Projects
This repository contains my projects for the [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).

## Project 1: Facial Keypoints Detection

This project is about defining and training a CNN to perform facial keypoint detection, and using computer vision techniques to transform images of faces. Facial keypoints (also called facial landmarks) mark important areas of the face: the eyes, corners of the mouth, the nose, etc. These keypoints are relevant for a variety of tasks, such as face filters, emotion recognition, pose recognition, etc. The model output is 68 keypoints, with coordinates (x, y), for that face, which I used later to add filters on the detected faces in an input live stream from a camera, running only on the CPU not the GPU without any delay in the feed.

Some results from my facial keypoint detection system:

<img src="https://github.com/RowanHisham/ComputerVision-nanodegree-Projects/blob/master/Images/facialKeypoints.jpg" alt="alt text" width="500" height="350">

###### USE KEYBOARD BUTTONS TO CHANGE FILTERS IN [LivePrediction.py](https://github.com/RowanHisham/ComputerVision-nanodegree-Projects/blob/master/CVND-P1-FacialKeypointsDetecion/LivePrediction.py) SCRIPT

<img src="https://github.com/RowanHisham/ComputerVision-nanodegree-Projects/blob/master/Images/facialKeypoints2.png" alt="alt text" width="500" height="350">

## Project 2: Image Captioning
In this project, I design and train a CNN-RNN (Convolutional Neural Network Encoder - Recurrent Neural Network Decoder) model for automatically generating image captions. The network is trained on the Microsoft Common Objects in Context (MS COCO) dataset. Trained only on 1 epoch the model reached an average loss= 2.007 and preplixity= 7.4. The model structure is described in the following image:

<img src="https://github.com/RowanHisham/ComputerVision-nanodegree-Projects/blob/master/Images/imageCaptioning6.png" alt="alt text" width="800" height="300">

<img src="https://github.com/RowanHisham/ComputerVision-nanodegree-Projects/blob/master/Images/imageCaptioning4.png" alt="alt text" width="900" height="50">


Some results from my image captioning model:

<img src="https://github.com/RowanHisham/ComputerVision-nanodegree-Projects/blob/master/Images/imageCaptioning3.png" alt="alt text" width="300" height="250"><img src="https://github.com/RowanHisham/ComputerVision-nanodegree-Projects/blob/master/Images/imageCaptioning2.png" alt="alt text" width="300" height="250"><img src="https://github.com/RowanHisham/ComputerVision-nanodegree-Projects/blob/master/Images/imageCaptioning1.png" alt="alt text" width="300" height="250">


