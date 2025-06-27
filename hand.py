import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3

# Load trained model (use your own .h5 model file path)
model = tf.keras.models.load_model('model/sign_language_model.h5')

# Label map for prediction (update this according to your dataset)
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    x, y, c = frame.shape
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(framergb)
    class_name = ''

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            x_min = y_min = 9999
            x_max = y_max = 0
            for lm in hand_landmarks.landmark:
                x_val, y_val = int(lm.x * x), int(lm.y * y)
                x_min = min(x_min, x_val)
                y_min = min(y_min, y_val)
                x_max = max(x_max, x_val)
                y_max = max(y_max, y_val)

            # Crop and preprocess hand ROI
            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue
            roi = cv2.resize(roi, (64, 64))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            roi = roi.reshape(1, 64, 64, 1) / 255.0

            # Predict
            predictions = model.predict(roi)
            class_id = np.argmax(predictions)
            class_name = labels[class_id]

            # Show prediction
            cv2.putText(frame, class_name, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Speak the output
            engine.say(class_name)
            engine.runAndWait()

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
