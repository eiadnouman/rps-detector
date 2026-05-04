# Hand Gesture Rock-Paper-Scissors Detector

A deep learning approach to Rock-Paper-Scissors gesture recognition using MediaPipe hand landmarks and EfficientNetB0 transfer learning.

## Overview

This project uses computer vision to detect and classify hand gestures for the Rock-Paper-Scissors game. It combines:
- **MediaPipe**: For accurate hand landmark detection
- **EfficientNetB0**: Pre-trained model with transfer learning for classification
- **Two-phase training**: Head-only training followed by fine-tuning for optimal performance

## Features

- Real-time hand gesture recognition
- High accuracy classification (Rock, Paper, Scissors)
- Transfer learning for efficient training
- Data augmentation for robustness
- Two-phase training strategy

## Project Structure

```
├── rsp-notebook.ipynb          # Main training notebook with full pipeline
├── load_model_helper.py        # Model loading utilities for Keras 3 compatibility
├── rps_best.keras              # Trained model weights
└── hand_landmarker.task        # MediaPipe hand detection model
```

## Requirements

- Python 3.8+
- TensorFlow/Keras
- MediaPipe
- OpenCV
- NumPy
- scikit-learn
- matplotlib
- seaborn

## Installation

```bash
pip install tensorflow mediapipe opencv-python numpy scikit-learn matplotlib seaborn
```

## Usage

### Training

Run the Jupyter notebook for full training pipeline:
```bash
jupyter notebook rsp-notebook.ipynb
```

### Loading Trained Model

Use the helper module to load the pre-trained model:
```python
from load_model_helper import load_rps_model

model = load_rps_model('rps_best.keras')
# Now use model for inference
predictions = model.predict(your_image)
```

The `load_model_helper.py` handles:
- Rebuilding the EfficientNetB0 architecture
- Loading weights from the .keras file (Keras 3 format)
- Proper initialization for inference

### Inference

Run the real-time detection script:
```bash
python RPS_Model.py
```

Controls:
- **SPACE**: Start/next round
- **Q**: Quit
- **R**: Reset scores

## Model Architecture

The model uses:
- **Base**: EfficientNetB0 (pre-trained on ImageNet)
- **Custom head**:
  - Global Average Pooling
  - Batch Normalization
  - Dense(256) + Dropout(0.4)
  - Dense(128) + Dropout(0.3)
  - Dense(3) with softmax (for 3 classes)

## Training Results

- Two-phase training approach
- Early stopping and learning rate reduction for optimization
- Model checkpointing for best weights
- Confusion matrix and classification reports generated

## Dataset

Uses the Rock-Paper-Scissors dataset from Kaggle:
- Training samples
- Validation samples
- Test samples

## License

Open source

## Author

Eiad Nouman
