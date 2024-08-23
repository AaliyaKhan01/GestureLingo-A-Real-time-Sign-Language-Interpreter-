#Code File 1: datacollection.py

import cv2
import mediapipe as mp
import os
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
# Initialize video capture
cap = cv2.VideoCapture(0)

# Create a directory to save images
save_dir = "captured_gestures"
os.makedirs(save_dir, exist_ok=True)

# Initialize list of letters to capture
alphabet = "OPQRSTUVWXYZ"
current_letter_index = 0
capture_count = 10  # Number of images to capture per letter

# Define a global variable to store the current frame
current_frame = None

# Define a function to handle mouse click events
def on_mouse_click(event, x, y, flags, param):
    global current_letter_index, current_frame, capture_count
    if event == cv2.EVENT_LBUTTONDOWN:
        if current_letter_index < len(alphabet):
            letter = alphabet[current_letter_index]
            letter_dir = os.path.join(save_dir, letter)
            os.makedirs(letter_dir, exist_ok=True)
            for i in range(capture_count):
                file_name = os.path.join(letter_dir, f"{letter}_{i}.png")
                if current_frame is not None:
                    cv2.imwrite(file_name, current_frame)
                    print(f"Captured {letter} {i+1}/{capture_count}")
                else:
                    print("No frame to capture")
            current_letter_index += 1
cv2.namedWindow('Capture Gestures')
cv2.setMouseCallback('Capture Gestures', on_mouse_click)
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

    # Display the current letter being captured
    if current_letter_index < len(alphabet):
        letter = alphabet[current_letter_index]
        cv2.putText(frame, f'Capture: {letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Store the current frame
    current_frame = frame.copy()


  
  # Show the frame with hand landmarks
    cv2.imshow('Capture Gestures', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
