import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
import mediapipe as mp
import os
import csv
import json
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sign_detector import SignDetectorClass

class SignLanguageSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language System")
        self.root.geometry("700x600")  # Increased width to accommodate wider text box

        # Configure grid weights for better resizing
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Main container frame
        main_frame = tk.Frame(root)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        main_frame.grid_columnconfigure(0, weight=1)

        # Title
        title_frame = tk.Frame(main_frame)
        title_frame.grid(row=0, column=0, pady=(0, 15))
        tk.Label(title_frame, text="Sign Language System", 
                font=("Arial", 24, "bold")).pack()

        # Buttons frame
        button_frame = tk.Frame(main_frame)
        button_frame.grid(row=1, column=0, pady=(0, 15))

        # Button style
        button_style = {
            'font': ('Arial', 12),
            'width': 15,
            'height': 1,
            'bg': '#4a90e2',
            'fg': 'white',
            'relief': 'raised',
            'padx': 10,
            'pady': 5
        }

        self.collect_btn = tk.Button(button_frame, text="Collect Data", 
                                   command=self.collect_data, **button_style)
        self.collect_btn.pack(pady=5)

        self.train_btn = tk.Button(button_frame, text="Train Model", 
                                  command=self.train_model, **button_style)
        self.train_btn.pack(pady=5)

        self.detect_btn = tk.Button(button_frame, text="Start Detection", 
                                  command=self.start_detection, **button_style)
        self.detect_btn.pack(pady=5)

        # Output frame
        output_frame = tk.Frame(main_frame)
        output_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        output_frame.grid_columnconfigure(0, weight=1)
        output_frame.grid_rowconfigure(1, weight=1)

        # Label for the text box
        tk.Label(output_frame, text="Detected Signs:", 
                font=('Arial', 14, 'bold')).grid(row=0, column=0, sticky="w", pady=(0, 5))

        # Output text box with better formatting
        self.output_box = tk.Text(output_frame, 
                               height=8,
                               width=55,   # Increased width for longer text
                               font=('Arial', 14),
                               bg='lightyellow',
                               wrap=tk.WORD,
                               state='disabled',
                               padx=10,
                               pady=10)
        self.output_box.grid(row=1, column=0, sticky="nsew")

        # Scrollbar for the text box
        scrollbar = tk.Scrollbar(output_frame, command=self.output_box.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.output_box.config(yscrollcommand=scrollbar.set)

        # Clear button with matching theme
        clear_btn = tk.Button(main_frame, 
                            text="Clear Text", 
                            command=self.clear_text,
                            font=('Arial', 12),
                            width=12,
                            height=1,
                            bg='#e74c3c',
                            fg='white',
                            relief='raised',
                            padx=10,
                            pady=5)
        clear_btn.grid(row=3, column=0, pady=(0, 10))

        # Store detector instance
        self.detector = None

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_output(self, text):
        """Update the output text box with smooth scrolling."""
        self.output_box.config(state='normal')
        self.output_box.delete(1.0, tk.END)
        self.output_box.insert(tk.END, text)
        self.output_box.config(state='disabled')
        self.output_box.see(tk.END)
        self.root.update()

    def collect_data(self):
        """Collect training data for a new sign."""
        label = simpledialog.askstring("Input", "Enter label for the gesture:")
        if not label:
            messagebox.showwarning("Warning", "No label entered. Data collection aborted.")
            return

        # Create dataset directory if it doesn't exist
        os.makedirs("dataset", exist_ok=True)
        csv_file = os.path.join("dataset", f"{label}_landmarks.csv")
        
        # Show instructions
        messagebox.showinfo("Instructions", 
                          "Make the sign in front of the camera.\n"
                          "Hold each position for a few seconds.\n"
                          "Press Q when done collecting data.")
        
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]])

            cap = cv2.VideoCapture(0)
            hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, 
                            hand_landmarks, 
                            mp.solutions.hands.HAND_CONNECTIONS
                        )
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        writer.writerow(landmarks)
                        frame_count += 1

                cv2.putText(frame, f"Collecting data for: {label}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Frames collected: {frame_count}", 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "Press Q to stop", 
                          (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Collecting Data", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            
            if frame_count > 0:
                messagebox.showinfo("Success", 
                                  f"Data collection for '{label}' completed!\n"
                                  f"Collected {frame_count} frames.")
            else:
                messagebox.showwarning("Warning", 
                                     "No data was collected. Please try again and "
                                     "make sure your hand is visible to the camera.")

    def train_model(self):
        """Train the model on collected data."""
        try:
            # Check if dataset directory exists
            if not os.path.exists("dataset"):
                messagebox.showerror("Error", "No dataset found. Please collect data first.")
                return

            # Check if there are any CSV files
            csv_files = [f for f in os.listdir("dataset") if f.endswith("_landmarks.csv")]
            if not csv_files:
                messagebox.showerror("Error", "No training data found. Please collect data first.")
                return

            # Prepare data
            X = []
            y = []
            labels = []
            
            # Load all CSV files from dataset directory
            for file in csv_files:
                label = file.split("_")[0]
                if label not in labels:
                    labels.append(label)
                
                try:
                    data = np.loadtxt(os.path.join("dataset", file), 
                                    delimiter=",", skiprows=1)
                    if len(data.shape) == 1:  # If only one sample, reshape it
                        data = data.reshape(1, -1)
                    X.extend(data)
                    y.extend([labels.index(label)] * len(data))
                except Exception as e:
                    messagebox.showerror("Error", f"Error loading {file}: {str(e)}")
                    return

            if not X:
                messagebox.showerror("Error", "No valid data found in the dataset.")
                return

            X = np.array(X)
            y = to_categorical(y)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Create and train model
            model = Sequential([
                Dense(128, activation='relu', input_shape=(63,)),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(len(labels), activation='softmax')
            ])

            model.compile(optimizer='adam', 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])

            # Create models directory
            os.makedirs("models", exist_ok=True)

            # Train the model
            history = model.fit(X_train, y_train, 
                              epochs=50, 
                              batch_size=32, 
                              validation_split=0.2,
                              verbose=1)

            # Save model
            model.save("models/sign_language_model.h5")
            
            # Create and save label mapping
            label_mapping = {label: idx for idx, label in enumerate(labels)}
            with open("models/signs_mapping.json", 'w') as f:
                json.dump(label_mapping, f, indent=4)

            # Show training results
            val_accuracy = history.history['val_accuracy'][-1]
            messagebox.showinfo("Success", 
                              f"Model training completed!\n"
                              f"Validation accuracy: {val_accuracy:.2%}\n"
                              f"Number of signs: {len(labels)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")

    def start_detection(self):
        """Start sign detection."""
        try:
            # Check if model exists
            if not os.path.exists("models/sign_language_model.h5"):
                messagebox.showerror("Error", "Model not found. Please train the model first.")
                return
                
            if not os.path.exists("models/signs_mapping.json"):
                messagebox.showerror("Error", "Signs mapping not found. Please train the model first.")
                return

            if self.detector:
                self.stop_detection()
                return

            self.detector = SignDetectorClass(self.update_output, self.root)
            self.detector.start_detection()
            self.detect_btn.config(text="Stop Detection")
            self.collect_btn.config(state="disabled")
            self.train_btn.config(state="disabled")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.stop_detection()

    def stop_detection(self):
        """Stop detection and reset UI."""
        if self.detector:
            self.detector.stop_detection()
            self.detector = None
            self.detect_btn.config(text="Start Detection")
            self.collect_btn.config(state="normal")
            self.train_btn.config(state="normal")

    def clear_text(self):
        """Clear the output text."""
        self.update_output("")
        if self.detector:
            self.detector.sentence = ""

    def on_closing(self):
        """Handle window closing."""
        if self.detector:
            self.stop_detection()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = SignLanguageSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main() 