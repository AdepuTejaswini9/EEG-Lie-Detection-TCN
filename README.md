# 🧠 Deep Learning-Based Lie Detection from Raw Multichannel EEG Signals Using Temporal Convolutional Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=flat-square)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.970-success?style=flat-square)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley-orange?style=flat-square)](https://data.mendeley.com/datasets/5gzxb2bzs2/2)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)


> A deep learning framework for EEG-based lie detection using a pure one-dimensional Temporal Convolutional Network (TCN) trained directly on raw multichannel EEG signals.

---

## 📌 Overview

Lie detection using electroencephalogram (EEG) signals has attracted considerable attention due to the limitations of traditional polygraph-based approaches. This project proposes a pure 1D Temporal Convolutional Network (TCN) that learns directly from raw EEG signals without handcrafted feature extraction, frequency-domain transformation, or wavelet decomposition.

The proposed model is compared with CNN, LSTM, and BiLSTM architectures using subject-wise evaluation.

### Key Highlights

* ✅ Pure 1D Temporal Convolutional Network (TCN)
* ✅ End-to-end learning from raw EEG signals
* ✅ Subject-wise evaluation
* ✅ Best Accuracy: **97%**
* ✅ ROC-AUC: **0.970**
* ✅ Outperforms CNN, LSTM, and BiLSTM

---

## 📊 Results

| Model              | Accuracy |
| ------------------ | -------- |
| CNN                | 62%      |
| LSTM               | 55%      |
| BiLSTM             | 56%      |
| **TCN (Proposed)** | **97%**  |

### Subject-wise Performance

| Subject   | Accuracy |
| --------- | -------- |
| Subject 1 | 94%      |
| Subject 2 | 97%      |
| Subject 3 | 96%      |

---

## 📂 Dataset

This project uses a publicly available EEG lie detection dataset.

Dataset Link:

https://data.mendeley.com/datasets/5gzxb2bzs2/2

### Dataset Characteristics

| Property     | Details             |
| ------------ | ------------------- |
| Data Type    | EEG Signals         |
| Classes      | Truthful, Deceptive |
| Subjects     | 3                   |
| Input        | Raw EEG             |
| Evaluation   | Subject-wise        |
| Segmentation | Sliding Window      |

---

## 📁 Project Structure

```text
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
```

---

## 🏗️ Model Architecture

The proposed architecture is a pure one-dimensional Temporal Convolutional Network (TCN).

Input EEG Window

↓

Dilated Conv1D

↓

Residual Block

↓

Batch Normalization

↓

Dropout

↓

Residual Block

↓

Global Average Pooling

↓

Dense Layer

↓

Sigmoid Output

↓

Truth / Lie

### Architecture Components

* Conv1D Layers
* Dilated Convolutions
* Residual Connections
* Batch Normalization
* Dropout
* Global Average Pooling
* Dense Layer

---

## ⚙️ Preprocessing Pipeline

### Applied

* Sliding Window Segmentation
* Overlapping Windows
* Subject-wise Train/Test Split

### Not Applied

* Band-pass Filtering
* Artifact Removal
* Wavelet Transform
* Handcrafted Feature Extraction
* Frequency-Domain Analysis

---

## 🏋️ Training Configuration

| Hyperparameter | Value                |
| -------------- | -------------------- |
| Optimizer      | Adam                 |
| Loss Function  | Binary Cross-Entropy |
| Batch Size     | 32                   |
| Epochs         | 30                   |
| Learning Rate  | 0.00025 – 0.001      |
| Patience       | 5                    |

---

## 📈 Evaluation Metrics

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC Curve
* AUC

### ROC-AUC

| Model | AUC   |
| ----- | ----- |
| TCN   | 0.970 |

---

## 📄 Research Paper

**Title:**

Deep Learning-Based Lie Detection from Raw Multichannel EEG Signals Using Temporal Convolutional Networks

**Status:** Under Review / Not Yet Published

The manuscript will be added after publication.

---

## 🔭 Future Work

* Cross-subject validation
* Real-time EEG lie detection
* Transformer-based architectures
* Hybrid TCN-Attention models
* Wearable EEG integration
* Forensic decision-support systems

---

## 👩‍💻 Author

**Adepu Tejaswini**

* LinkedIn: Add your LinkedIn URL
* GitHub: Add your GitHub URL
* Email: [adeputejaswini7@gmail.com](mailto:adeputejaswini7@gmail.com)

---

⭐ If you found this project useful, consider giving it a star.
