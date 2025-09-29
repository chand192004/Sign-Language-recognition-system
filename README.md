# Sign Language Recognition System 🤟

A real-time sign language recognition system using deep learning and computer vision to detect and interpret sign language gestures.

## 📋 Table of Contents
- Overview
- Features
- Project Structure
- Installation
- Usage
- Dataset
- Model Architecture
- Requirements
- Contributing
- License

## 🎯 Overview

This project implements a machine learning-based sign language recognition system that can detect and interpret sign language gestures in real-time using a webcam. The system uses computer vision techniques and deep learning models to recognize hand gestures and translate them into text.

## ✨ Features

- Real-time sign language gesture recognition
- Custom dataset collection and preprocessing
- Deep learning model training pipeline
- Webcam integration for live detection
- Support for multiple sign language gestures
- Model performance visualization and metrics

## 📁 Project Structure

```
Sign-Language-recognition-system/
│
├── __pycache__/          # Python cache files
├── dataset/              # Training and testing datasets
├── models/               # Saved trained models
├── venv/                 # Virtual environment
│
├── data_collector.py     # Script to collect training data
├── main.py               # Main application entry point
├── model_trainer.py      # Model training script
├── sign_detector.py      # Real-time sign detection
├── signs_mapping.json    # Mapping of signs to labels
├── requirements.txt      # Project dependencies
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam (for real-time detection)
- pip package manager

### Steps

1. **Clone the repository**
```bash
git clone https://github.com/chand192004/Sign-Language-recognition-system.git
cd Sign-Language-recognition-system
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 💻 Usage

### 1. Collect Training Data

First, collect gesture data for training:

```bash
python data_collector.py
```

Follow the on-screen instructions to record different sign language gestures.

### 2. Train the Model

Train the recognition model using collected data:

```bash
python model_trainer.py
```

The trained model will be saved in the `models/` directory.

### 3. Run Real-time Detection

Start the sign language detection system:

```bash
python main.py
```

Or use the dedicated sign detector:

```bash
python sign_detector.py
```

Press 'q' to quit the application.

## 📊 Dataset

The dataset consists of:
- Hand gesture images/videos
- Multiple sign language gestures
- Training and validation splits
- Augmented data for better model generalization

The `signs_mapping.json` file contains the mapping between gesture classes and their labels.

## 🧠 Model Architecture

The system uses a deep learning model (likely CNN-based) for gesture recognition:
- Input: Video frames or image sequences
- Feature extraction layers
- Classification layers
- Output: Predicted sign language gesture

Model files are saved in `.h5` format in the `models/` directory.

## 📦 Requirements

Key dependencies include:
- TensorFlow/Keras - Deep learning framework
- OpenCV - Computer vision library
- NumPy - Numerical computing
- MediaPipe/CVZone - Hand tracking (if applicable)

See `requirements.txt` for complete list of dependencies.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👨‍💻 Author

**Chand** - [@chand192004](https://github.com/chand192004)


