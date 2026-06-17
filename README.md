# 🧠 Deep Learning-Based Lie Detection from Raw Multichannel EEG Signals Using Temporal Convolutional Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=flat-square)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.970-success?style=flat-square)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley-orange?style=flat-square)](https://data.mendeley.com/datasets/5gzxb2bzs2/2)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)

A deep learning framework for EEG-based lie detection using a pure one-dimensional Temporal Convolutional Network (TCN) trained directly on raw multichannel EEG signals.

📌 Table of Contents
Overview
Results
Dataset
Project Structure
Getting Started
Model Architecture
Preprocessing Pipeline
Training Configuration
Evaluation
Future Work
References
🔍 Overview

Lie detection using electroencephalogram (EEG) signals has gained considerable attention due to the limitations of traditional polygraph-based approaches. EEG provides direct access to neural activity, enabling the analysis of brain responses associated with truthful and deceptive behavior.

This project proposes a pure 1D Temporal Convolutional Network (TCN) for EEG-based lie detection. Unlike conventional approaches that rely on handcrafted features, frequency-domain transformations, or wavelet decomposition, the proposed model learns directly from raw multichannel EEG signals using an end-to-end deep learning pipeline.

The proposed TCN is compared against CNN, LSTM, and BiLSTM architectures using a subject-wise evaluation protocol.

Key Highlights
✅ Pure 1D Temporal Convolutional Network (TCN)
✅ End-to-end learning from raw EEG signals
✅ Raw multichannel EEG input
✅ Subject-wise evaluation
✅ Best Accuracy: 97%
✅ ROC-AUC: 0.970
✅ Outperforms CNN, LSTM, and BiLSTM baselines
📊 Results
Model	Accuracy
CNN	62%
LSTM	55%
BiLSTM	56%
TCN (Proposed)	97%
Subject-wise Performance
Subject	Accuracy
Subject 1	94%
Subject 2	97%
Subject 3	96%
Performance Improvement
Comparison	Improvement
TCN vs CNN	+35%
TCN vs LSTM	+42%
TCN vs BiLSTM	+41%

📁 Confusion matrix, ROC curve, and training accuracy plots are available in the results/ folder.

📂 Dataset

This project uses a publicly available EEG lie detection dataset from Mendeley Data.

Dataset Link

https://data.mendeley.com/datasets/5gzxb2bzs2/2

Dataset Characteristics
Property	Details
Data Type	EEG Signals
Classes	Truthful, Deceptive
Subjects	3
Input Type	Raw EEG
Evaluation	Subject-wise
Segmentation	Sliding Window
Dataset Split
Split	Proportion
Training	80%
Testing	20%
📁 Project Structure
EEG-Lie-Detection-TCN/
│
├── README.md
├── requirements.txt
│
├── Lie_Detection.ipynb
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── accuracy_vs_epochs.png
│
├── figures/
│   ├── workflow.png
│   ├── tcn_architecture.png
│
└── LICENSE
🚀 Getting Started
Prerequisites
Python 3.8+
PyTorch 2.0+
Jupyter Notebook
NumPy
Pandas
Matplotlib
Scikit-learn
Clone the Repository
git clone https://github.com/your-username/EEG-Lie-Detection-TCN.git
cd EEG-Lie-Detection-TCN
Install Dependencies
pip install -r requirements.txt
Run the Notebook
jupyter notebook

Open:

Lie_Detection.ipynb

and execute all cells.

🏗️ Model Architecture

The proposed model is a pure one-dimensional Temporal Convolutional Network (TCN) designed for sequence modeling of EEG signals.

Input EEG Window
        │
        ▼
Dilated Conv1D
        │
Residual Block
        │
Batch Normalization
        │
Dropout
        │
Residual Block
        │
Global Average Pooling
        │
Dense Layer
        │
Sigmoid Output
        │
Truth / Lie
Architecture Components
Conv1D Layers
Dilated Convolutions
Residual Connections
Batch Normalization
Dropout Regularization
Global Average Pooling
Dense Classification Layer
⚙️ Preprocessing Pipeline
Applied
Sliding Window Segmentation
Overlapping Window Generation
Subject-wise Train/Test Split
Not Applied
Band-pass Filtering
Artifact Removal (ICA/EOG)
Wavelet Transform
Handcrafted Feature Extraction
Frequency-Domain Analysis

The model learns directly from raw EEG time-series windows.

🏋️ Training Configuration
Hyperparameter	Value
Optimizer	Adam
Loss Function	Binary Cross-Entropy
Batch Size	32
Epochs	30
Learning Rate	0.00025 – 0.001
Patience	5
Regularization Techniques
Batch Normalization
Dropout
Residual Connections
📈 Evaluation

The model was evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC Curve
Area Under Curve (AUC)
ROC-AUC
Model	AUC
TCN	0.970
📄 Research Paper

Title:

Deep Learning-Based Lie Detection from Raw Multichannel EEG Signals Using Temporal Convolutional Networks

Status: Under Review / Not Yet Published

The manuscript will be added after publication.

🔭 Future Work
Cross-subject evaluation
Real-time EEG lie detection
Transformer-based architectures
Hybrid TCN-Attention models
Portable EEG device integration
Forensic and security applications
📚 References
S. Bai, J. Z. Kolter, and V. Koltun, An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, 2018.
H. A. H. Mahmoud, A Deep Learning Model for EEG-Based Lie Detection Test Using Spatial and Temporal Aspects, 2022.
N. Farah et al., Neurophysiological Approaches to Lie Detection, 2025.
D. Avola et al., Bi-GRU Based Deception Detection Using EEG Signals, 2025.
👩‍💻 Author

Adepu Tejaswini

GitHub: https://github.com/your-username
LinkedIn: https://linkedin.com/in/your-profile
Email: adeputejaswini7@gmail.com
