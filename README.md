# Facial-Emotion-Recognition-Using-Explainable-AI

This notebook demonstrates a complete workflow for facial emotion recognition using deep learning and explainable AI techniques, utilizing a Convolutional Neural Network (CNN) for emotion detection and LIME for model interpretability.

---

## Overview

The notebook is organized into the following main sections:

### 1. Data Preparation

- The `load_kdef_data` function prepares data from the KDEF (Karolinska Directed Emotional Faces) dataset, which contains images organized by emotion categories.
- Data is split into training and validation sets, with image augmentation and normalization handled by TensorFlow's `ImageDataGenerator`.

### 2. Model Building and Training

- A CNN is built using TensorFlow’s Keras API, incorporating convolutional and pooling layers along with batch normalization and dropout for regularization.
- The model is trained on the prepared data with categorical cross-entropy as the loss function and accuracy as the primary performance metric.

### 3. Model Evaluation

- The trained model is evaluated on the validation set, with accuracy, precision, recall, and F1 scores calculated.
- A classification report provides detailed performance metrics for each emotion class.
- Training and validation accuracy and loss are visualized across epochs.

### 4. Prediction and Visualization

- The `predict_and_display_results` function predicts emotions on test images, displaying actual and predicted labels for each image to demonstrate the model’s performance.

### 5. Explainability with LIME

- Using the LIME (Local Interpretable Model-agnostic Explanations) library, the notebook visualizes model prediction explanations on sample images.
- Explanation results highlight the image regions influencing predictions, providing insights into how the model interprets different emotional expressions.

### 6. Sample Code Execution

The notebook installs necessary libraries, processes the KDEF dataset, trains the CNN model, and visualizes predictions and LIME explanations.

---

## Requirements

To run this notebook, you'll need:

- **Dataset**: KDEF (Karolinska Directed Emotional Faces) or another emotion-labeled image dataset, organized by emotion categories.
- **Environment**: This notebook assumes execution on Google Colab, but it should work in any Python environment after setting up the paths and installing the required libraries.

### Required Libraries

```python
!pip install tensorflow matplotlib scikit-learn lime opencv-python
