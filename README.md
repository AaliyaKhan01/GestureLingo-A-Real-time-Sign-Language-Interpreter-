# GestureLingo-A-Real-time-Sign-Language-Interpreter
Project Objectives

Develop a Real-Time Gesture Recognition System: Create a system that can recognize and interpret hand gestures from live video feed using a webcam.
Capture and Save Gesture Images: Implement functionality to capture images of hand gestures, saving them for use in training the recognition model.
Train a Gesture Recognition Model: Preprocess and store captured gesture images to train a model for accurate gesture recognition.
Display Recognized Gestures: Implement real-time recognition of gestures and display the results on the video feed.

Introduction

This project implements a real-time gesture recognition system using Python, OpenCV, and MediaPipe. It captures and recognizes hand gestures through a webcam, translating them into corresponding letters from the American Sign Language (ASL) alphabet.

How It Works
Data Collection (Code 1: datacollection.py):

Purpose: Captures and saves images of hand gestures.
How It Works:
Initializes the webcam and MediaPipe Hands to detect hand landmarks.
Captures frames from the webcam, processes them to detect gestures, and saves images when the user clicks on the video feed.
Images are saved in directories named after each gesture letter, creating a dataset for training.

Gesture Recognition (Code 2: translator.py):

Purpose: Recognizes hand gestures in real-time.

How It Works:
Loads the captured gesture images and preprocesses them for recognition.
Continuously captures frames from the webcam, processes them to detect hand landmarks, and extracts the hand region.
Compares the extracted gesture with stored gestures using cosine similarity to determine the closest match.
Displays the recognized gesture on the video feed in real-time.

Background Processing

Image Processing:
Images are converted to grayscale, resized, and normalized to ensure consistency and improve recognition accuracy.

Recognition Algorithm:
Uses cosine similarity to compare captured hand gestures with stored images, identifying the closest match.

This system enables real-time gesture recognition, making it practical for applications requiring gesture-based input.
