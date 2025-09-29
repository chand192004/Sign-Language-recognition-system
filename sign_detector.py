import cv2
import mediapipe as mp
import numpy as np
import threading
import json
from tensorflow.keras.models import load_model

class SignDetectorClass:
    def __init__(self, update_callback, root):
        self.update_callback = update_callback
        self.root = root
        self.running = False
        self.thread = None
        self.sentence = ""
        self.last_prediction = None
        self.prediction_count = 0
        self.PREDICTION_THRESHOLD = 10  # Number of consistent predictions needed
        self.last_update_time = 0
        self.UPDATE_INTERVAL = 0.1  # Minimum time between updates in seconds

        # Load the model
        self.model = load_model("models/sign_language_model.h5")

        # Load the signs mapping
        with open("models/signs_mapping.json", 'r') as f:
            self.signs_mapping = json.load(f)
            # Create reverse mapping (index to label)
            self.index_to_label = {str(v): k for k, v in self.signs_mapping.items()}

        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self):
        """Process video frames and detect signs."""
        cap = cv2.VideoCapture(0)
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame and detect hands
            results = self.hands.process(rgb_frame)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS
                    )

                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    # Make prediction
                    prediction = self.model.predict(
                        np.array(landmarks).reshape(1, -1),
                        verbose=0
                    )
                    predicted_index = str(np.argmax(prediction[0]))
                    confidence = prediction[0][int(predicted_index)]

                    if confidence > 0.95:  # High confidence threshold
                        predicted_sign = self.index_to_label.get(predicted_index, "Unknown")
                        
                        # Check if this is the same as the last prediction
                        if predicted_sign == self.last_prediction:
                            self.prediction_count += 1
                        else:
                            self.prediction_count = 1
                            self.last_prediction = predicted_sign

                        # If we have enough consistent predictions, update the sentence
                        if self.prediction_count >= self.PREDICTION_THRESHOLD:
                            current_time = cv2.getTickCount() / cv2.getTickFrequency()
                            if current_time - self.last_update_time >= self.UPDATE_INTERVAL:
                                if not self.sentence or self.sentence.split()[-1] != predicted_sign:
                                    self.sentence += f" {predicted_sign}"
                                    self.update_callback(self.sentence.strip())
                                    self.prediction_count = 0  # Reset counter
                                    self.last_update_time = current_time

                        # Display prediction and confidence
                        cv2.putText(
                            frame,
                            f"Sign: {predicted_sign} ({confidence:.2%})",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                    
            # Display the frame
            cv2.imshow("Sign Language Detection", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Clean up
        cap.release()
        cv2.destroyAllWindows()

    def start_detection(self):
        """Start the detection process in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self.process_frame)
        self.thread.start()

    def stop_detection(self):
        """Stop the detection process."""
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None
