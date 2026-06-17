# 🧠 Deep Learning-Based Lie Detection from Raw Multichannel EEG Signals Using Temporal Convolutional Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square\&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square\&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=flat-square)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.970-success?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-Mendeley-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)


A Temporal Convolutional Network (TCN)-based framework for EEG lie detection that classifies brain signals into Truthful and Deceptive responses directly from raw multichannel EEG recordings, achieving 97% accuracy and 0.970 ROC-AUC.

📌 Table of Contents
Overview
Results
Dataset
Project Structure
Getting Started
Model Architecture
Preprocessing Pipeline
Training
Evaluation
Tech Stack
Future Work
References
🔍 Overview

Lie detection using Electroencephalography (EEG) has emerged as a promising alternative to traditional polygraph-based approaches. EEG captures neural activity associated with cognitive processes involved in deception, providing a more objective measure of truthfulness.

This project proposes a Temporal Convolutional Network (TCN) for binary classification of EEG signals into truthful and deceptive responses. Unlike conventional approaches that rely on handcrafted features or frequency-domain transformations, the proposed model learns discriminative temporal patterns directly from raw EEG signals.

Key Highlights
✅ 97% classification accuracy
✅ ROC-AUC score of 0.970
✅ Binary classification (Truthful vs Deceptive)
✅ Direct learning from raw EEG signals
✅ No feature engineering required
✅ Outperforms CNN, LSTM, and Bi-LSTM models
📊 Results
Overall Performance
Model	Accuracy	Precision	Recall	F1-Score
TCN (Proposed)	97%	0.97	0.97	0.97
CNN	62%	0.63	0.62	0.62
LSTM	55%	0.55	0.55	0.54
Bi-LSTM	56%	0.58	0.56	0.53
Subject-wise Performance
Subject	Accuracy	Precision	Recall	F1-Score
Subject 1	0.94	0.94	0.94	0.94
Subject 2	0.97	0.98	0.97	0.97
Subject 3	0.96	0.96	0.96	0.95
📂 Dataset

A custom EEG dataset was created through a controlled lie detection experiment involving three participants.

Dataset Characteristics
Property	Details
Subjects	3
Classes	Truthful, Deceptive
Signal Type	Raw Multichannel EEG
Classification Type	Binary
Data Split	80% Training, 20% Testing
Segmentation	Sliding Window Approach
Data Preparation
Noise removal and data cleaning
Sliding window segmentation
Subject-wise train-test split
Balanced class distribution
Raw EEG signals used directly without feature extraction
📁 Project Structure
EEG-Lie-Detection-TCN/
│
├── README.md
├── requirements.txt
│
├── EEG_Lie_Detection_TCN.ipynb
│
├── models/
│   ├── tcn_model.pth
│   ├── cnn_model.pth
│   ├── lstm_model.pth
│   └── bilstm_model.pth
│
├── results/
│   ├── confusion_matrix_subject1.png
│   ├── confusion_matrix_subject2.png
│   ├── roc_curve.png
│   └── accuracy_curve.png
│
└── Project_Report.pdf
🚀 Getting Started
Prerequisites
Python 3.8+
PyTorch
NumPy
Pandas
Matplotlib
Scikit-learn
Clone Repository
git clone https://github.com/your-username/EEG-Lie-Detection-TCN.git

cd EEG-Lie-Detection-TCN
Install Dependencies
pip install -r requirements.txt
Run Training
python train.py
Evaluate Model
python evaluate.py
🏗️ Model Architecture

The proposed TCN architecture consists of:

Input EEG Signal
        │
        ▼
Dilated Conv1D Layer
        │
Batch Normalization
        │
ReLU Activation
        │
Dropout
        │
Residual Connection
        │
Dilated Conv1D Layer
        │
Batch Normalization
        │
Dropout
        │
Global Average Pooling
        │
Fully Connected Layer
        │
Sigmoid Output
        │
Truthful / Deceptive
Why TCN?
Captures long-range temporal dependencies
Parallel computation capability
Stable gradient flow using residual connections
Lower training complexity than recurrent models
Better performance on EEG time-series data
⚙️ Preprocessing Pipeline

The EEG recordings undergo the following preprocessing stages:

1. Data Cleaning
Removal of noisy segments
Removal of incomplete trials
2. Sliding Window Segmentation
Continuous EEG signals divided into overlapping windows
Preserves temporal information
3. Subject-wise Splitting
Training Data : 80%
Testing Data  : 20%
4. Model Input Preparation
EEG windows reshaped into one-dimensional sequences
Directly fed into deep learning models
🏋️ Training
Training Configuration
Hyperparameter	Value
Optimizer	Adam
Batch Size	32
Loss Function	Binary Cross Entropy
Architecture	TCN
Output Layer	Sigmoid
Classification	Binary
Regularization Techniques
Batch Normalization
Dropout
Residual Learning
📈 Evaluation

The model is evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC
Generated Outputs
Confusion Matrix
Accuracy Curves
ROC Curve
Classification Metrics
🛠️ Tech Stack
Category	Technology
Programming Language	Python
Deep Learning	PyTorch
Data Processing	NumPy, Pandas
Visualization	Matplotlib
Machine Learning	Scikit-learn
Development Environment	Jupyter Notebook
🔭 Future Work
 Cross-subject generalization
 Real-time lie detection system
 Transformer-based EEG architectures
 Attention-enhanced TCN models
 Integration with portable EEG devices
 Forensic and security decision-support systems
📚 References
Bai, S., Kolter, J.Z., Koltun, V. — Temporal Convolutional Networks for Sequence Modeling.
Mahmoud, H.A.H. — Deep Learning Model for EEG-Based Lie Detection.
Farah, N. et al. — Neurophysiological Approaches to Lie Detection.
Wen, X., Li, W. — Time Series Prediction Using LSTM-Based Architectures.
Avola, D. et al. — EEG-Based Deception Detection Using Deep Learning.
👩‍💻 Authors

Adepu Tejaswini
📧 adeputejaswini7@gmail.com

Boda Eramma

Mogili Keerthi

Dr. Sridhar Chintala
