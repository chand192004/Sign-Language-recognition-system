import cv2
import mediapipe as mp
import numpy as np
import os
from tkinter import messagebox, simpledialog
import json

class DataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,  # Lowered threshold for better detection
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create directories if they don't exist
        self.data_dir = "dataset"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Load or create signs mapping
        self.signs_file = "signs_mapping.json"
        if os.path.exists(self.signs_file):
            with open(self.signs_file, 'r') as f:
                self.signs_mapping = json.load(f)
        else:
            self.signs_mapping = {}

    def start_collection(self):
        # Get sign label from user
        sign = simpledialog.askstring("Input", "Enter the sign label (e.g., 'A', 'Hello'):")
        if not sign:
            return
            
        # Update signs mapping
        if sign not in self.signs_mapping:
            next_id = len(self.signs_mapping)
            self.signs_mapping[sign] = next_id
            with open(self.signs_file, 'w') as f:
                json.dump(self.signs_mapping, f)
        
        # Create directory for this sign
        sign_dir = os.path.join(self.data_dir, sign)
        if not os.path.exists(sign_dir):
            os.makedirs(sign_dir)
        
        # Try different camera indices
        cap = None
        for i in range(2):  # Try camera index 0 and 1
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"Successfully opened camera {i}")
                break
            else:
                print(f"Failed to open camera {i}")
        
        if not cap or not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera. Please check if your camera is connected and not in use by another application.")
            return
        
        # Get camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {frame_width}x{frame_height}")
        
        frame_count = len([name for name in os.listdir(sign_dir) if name.endswith('.npy')])
        print(f"Starting collection for sign '{sign}'. Press SPACEBAR to capture, 'q' to quit.")
        print(f"Current frame count: {frame_count}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Extract landmarks
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # Save landmarks when spacebar is pressed
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:  # Spacebar
                        landmark_file = os.path.join(sign_dir, f"frame_{frame_count}.npy")
                        np.save(landmark_file, landmarks)
                        frame_count += 1
                        print(f"Saved frame {frame_count} for sign {sign}")
                    elif key == ord('q'):
                        break
            else:
                # Display message when no hand is detected
                cv2.putText(display_frame, "No hand detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display frame count and instructions
            cv2.putText(display_frame, f"Frames: {frame_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE: Capture  Q: Quit", (10, frame_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Data Collection", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Success", f"Collected {frame_count} frames for sign {sign}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.start_collection() 