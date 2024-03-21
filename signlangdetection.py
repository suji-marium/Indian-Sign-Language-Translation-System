
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model_path = 'model15.keras'
model = tf.keras.models.load_model(model_path)

# Create a mapping between numerical classes and alphabet letters
class_to_alphabet = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K',
                    11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U',
                    21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# OpenCV setup
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    ret, frame = cap.read()

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform hand pose estimation
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame for each hand
            for landmark in hand_landmarks.landmark:
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # Resize the hand region to (100, 100) for model input
            hand_roi = frame
            resized_hand = cv2.resize(hand_roi, (100, 100))

            # Convert the hand region to grayscale
            gray_hand = cv2.cvtColor(resized_hand, cv2.COLOR_BGR2GRAY)

            # Normalize pixel values
            normalized_hand = gray_hand / 255.0

            # Perform prediction using your model
            prediction = model.predict(np.expand_dims(normalized_hand, axis=0))

            # Get the predicted class
            predicted_class = np.argmax(prediction)

            # Map the predicted class to the corresponding alphabet letter
            predicted_letter = class_to_alphabet.get(predicted_class, 'Unknown')

            # Display the result on the frame for each hand
            cv2.putText(frame, f'Predicted: {predicted_letter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Indian Sign Language Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()


