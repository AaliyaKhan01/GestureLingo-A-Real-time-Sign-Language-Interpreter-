#Code File 2: translator.py

import cv2
import numpy as np
import mediapipe as mp
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load captured gestures
gesture_dir = "captured_gestures"
gesture_images = {}
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
capture_count = 10  # Number of images captured per letter

# Preprocess and store gesture images
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 64))
    image = image.flatten()
    return normalize([image])[0]

for letter in alphabet:
    gesture_images[letter] = []
    letter_dir = os.path.join(gesture_dir, letter)
    for i in range(capture_count):
        file_path = os.path.join(letter_dir, f"{letter}_{i}.png")
        if os.path.exists(file_path):
            img = cv2.imread(file_path)
            if img is not None:
                gesture_images[letter].append(preprocess_image(img))
            else:
                print(f"Failed to load image for letter {letter} {i+1}")

# Initialize video capture
cap = cv2.VideoCapture(0)

def recognize_gesture(frame):
    # Extract hand region
    x_min = int(0)
    x_max = int(frame.shape[1])
    y_min = int(0)
    y_max = int(frame.shape[0])
    hand_region = frame[y_min:y_max, x_min:x_max]
    preprocessed_hand = preprocess_image(hand_region)

    # Compare with saved gestures
    best_match = "Unknown"
    best_similarity = -1
    for letter, stored_gestures in gesture_images.items():
        for stored_gesture in stored_gestures:
            similarity = cosine_similarity([preprocessed_hand], [stored_gesture])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = letter

    return best_match

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize the gesture
            gesture_letter = recognize_gesture(frame)
            print(f'Recognized Gesture: {gesture_letter}')  # Print to command prompt
            cv2.putText(frame, f'Gesture: {gesture_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the frame with hand landmarks
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
