import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tkinter import messagebox
import tensorflow as tf
import glob

class ModelTrainer:
    def __init__(self):
        self.model_dir = "models"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Load signs mapping
        try:
            with open("signs_mapping.json", 'r') as f:
                self.signs_mapping = json.load(f)
                print(f"Loaded signs mapping: {self.signs_mapping}")
                
            if len(self.signs_mapping) < 2:
                messagebox.showwarning("Warning", "You need at least 2 different signs to train the model.")
        except FileNotFoundError:
            messagebox.showerror("Error", "No signs mapping found. Please collect data first.")
            raise
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Signs mapping file is corrupted.")
            raise
            
        self.num_classes = len(self.signs_mapping)
        self.input_shape = 63  # 21 landmarks * 3 coordinates (x, y, z)
        print(f"Number of classes: {self.num_classes}")

    def load_data(self):
        X = []
        y = []
        
        total_samples = 0
        samples_per_class = {}
        
        for sign, label in self.signs_mapping.items():
            sign_dir = os.path.join("dataset", sign)
            if not os.path.exists(sign_dir):
                print(f"Warning: Directory not found for sign '{sign}'")
                continue
                
            npy_files = glob.glob(os.path.join(sign_dir, "*.npy"))
            samples_per_class[sign] = len(npy_files)
            total_samples += len(npy_files)
            
            print(f"Loading {len(npy_files)} samples for sign '{sign}'")
            
            for file in npy_files:
                try:
                    landmarks = np.load(file)
                    if landmarks.shape[0] != self.input_shape:
                        print(f"Warning: Skipping file {file} due to incorrect shape {landmarks.shape}")
                        continue
                    X.append(landmarks)
                    y.append(label)
                except Exception as e:
                    print(f"Error loading file {file}: {str(e)}")
        
        if total_samples == 0:
            raise ValueError("No training data found in the dataset directory")
            
        if total_samples < 20:  # Minimum recommended samples
            messagebox.showwarning("Warning", f"Only {total_samples} total samples found. It's recommended to have at least 20 samples per sign for better accuracy.")
            
        print("\nSamples per class:")
        for sign, count in samples_per_class.items():
            print(f"{sign}: {count} samples")
        
        return np.array(X), np.array(y)

    def create_model(self):
        print("\nCreating model architecture...")
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.input_shape,)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model

    def start_training(self):
        try:
            print("Starting the training process...")
            
            # Load and preprocess data
            print("\nLoading dataset...")
            X, y = self.load_data()
            
            print(f"\nTotal samples: {len(X)}")
            print(f"Input shape: {X.shape}")
            print(f"Output shape: {y.shape}")
            
            if len(X) == 0:
                messagebox.showerror("Error", "No training data found!")
                return
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            print(f"\nTraining samples: {len(X_train)}")
            print(f"Testing samples: {len(X_test)}")
            
            # Create and train model
            model = self.create_model()
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    os.path.join(self.model_dir, 'best_model.keras'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train
            print("\nStarting training...")
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            print("\nEvaluating model...")
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
            
            # Save the model
            print("\nSaving model...")
            model.save(os.path.join(self.model_dir, 'final_model.keras'))
            
            # Save the signs mapping with the model
            mapping_file = os.path.join(self.model_dir, 'signs_mapping.json')
            with open(mapping_file, 'w') as f:
                json.dump(self.signs_mapping, f)
            print(f"Saved signs mapping to {mapping_file}")
            
            messagebox.showinfo(
                "Training Complete",
                f"Model trained successfully!\nTest Accuracy: {test_accuracy:.2%}"
            )
            
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            print(f"Error: {error_msg}")
            messagebox.showerror("Error", error_msg)
            raise

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.start_training() 