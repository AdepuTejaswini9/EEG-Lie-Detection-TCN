# 🧠 Deep Learning-Based Lie Detection from Raw Multichannel EEG Signals Using Temporal Convolutional Networks

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square\&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square\&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-97%25-brightgreen?style=flat-square)
![ROC--AUC](https://img.shields.io/badge/ROC--AUC-0.970-success?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-LieWaves-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)

> A Temporal Convolutional Network (TCN)-based framework for EEG lie detection that classifies brain signals into **Truthful** and **Deceptive** responses directly from raw multichannel EEG recordings, achieving **97% accuracy** and **0.970 ROC-AUC**.

---

# 📌 Table of Contents

* Overview
* Results
* Dataset
* Project Structure
* Getting Started
* Model Architecture
* Preprocessing Pipeline
* Training
* Evaluation
* Tech Stack
* Future Work
* References
* Authors

---

# 🔍 Overview

Lie detection using Electroencephalography (EEG) has emerged as a promising alternative to traditional polygraph-based approaches. EEG captures neural activity associated with cognitive processes involved in deception, providing a more objective measure of truthfulness.

This project proposes a **Temporal Convolutional Network (TCN)** for binary classification of EEG signals into truthful and deceptive responses. Unlike conventional approaches that rely on handcrafted features or frequency-domain transformations, the proposed model learns discriminative temporal patterns directly from raw EEG signals.

## ✨ Key Highlights

* ✅ 97% Classification Accuracy
* ✅ ROC-AUC Score of 0.970
* ✅ Binary Classification (Truthful vs Deceptive)
* ✅ Direct Learning from Raw EEG Signals
* ✅ No Feature Engineering Required
* ✅ Outperforms CNN, LSTM, and Bi-LSTM Models

---

# 📊 Results

## Overall Performance

| Model              | Accuracy | Precision | Recall   | F1-Score |
| ------------------ | -------- | --------- | -------- | -------- |
| **TCN (Proposed)** | **97%**  | **0.97**  | **0.97** | **0.97** |
| CNN                | 62%      | 0.63      | 0.62     | 0.62     |
| LSTM               | 55%      | 0.55      | 0.55     | 0.54     |
| Bi-LSTM            | 56%      | 0.58      | 0.56     | 0.53     |

## Subject-wise Performance

| Subject   | Accuracy | Precision | Recall | F1-Score |
| --------- | -------- | --------- | ------ | -------- |
| Subject 1 | 0.94     | 0.94      | 0.94   | 0.94     |
| Subject 2 | 0.97     | 0.98      | 0.97   | 0.97     |
| Subject 3 | 0.96     | 0.96      | 0.96   | 0.95     |

---

# 📂 Dataset

This project utilizes the **LieWaves Dataset**, a publicly available EEG dataset specifically designed for deception detection research.

## Dataset Information

| Property        | Details                  |
| --------------- | ------------------------ |
| Dataset Name    | LieWaves                 |
| Source          | Mendeley Data            |
| Subjects        | 27                       |
| EEG Device      | Emotiv Insight           |
| Channels        | 5 (AF3, T7, Pz, T8, AF4) |
| Classes         | Truthful, Deceptive      |
| Signal Type     | Raw EEG Time Series      |
| Format          | CSV                      |
| Dataset Version | V2                       |

## EEG Channel Configuration

```text
AF3 ───── Frontal Region
T7  ───── Left Temporal Region
Pz  ───── Parietal Region
T8  ───── Right Temporal Region
AF4 ───── Frontal Region
```

## Download Dataset

https://data.mendeley.com/datasets/5gzxb2bzs2/2

---

# 📁 Project Structure

```text
EEG-Lie-Detection-TCN/
│
├── README.md
├── requirements.txt
├── EEG_Lie_Detection_TCN.ipynb
│
├── models/
│   ├── tcn_model.pth
│   ├── cnn_model.pth
│   ├── lstm_model.pth
│   └── bilstm_model.pth
│
├── results/
   ├── confusion_matrix_subject1.png
   ├── confusion_matrix_subject2.png
   ├── roc_curve.png
   └── accuracy_curve.png

```

---

# 🚀 Getting Started

## Prerequisites

* Python 3.8+
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## Clone Repository

```bash
git clone https://github.com/your-username/EEG-Lie-Detection-TCN.git

cd EEG-Lie-Detection-TCN
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python train.py
```

## Evaluate Model

```bash
python evaluate.py
```

---

# 🏗️ Model Architecture

```text
┌─────────────────────────────┐
│      Input EEG Signal       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Dilated Conv1D Layer      │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│    Batch Normalization      │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│      ReLU Activation        │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│          Dropout            │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│    Residual Connection      │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   Dilated Conv1D Layer      │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│    Batch Normalization      │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│          Dropout            │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Global Average Pooling     │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│   Fully Connected Layer     │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│       Sigmoid Output        │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Truthful / Deceptive       │
└─────────────────────────────┘
```

---

# ⚙️ Preprocessing Pipeline

1. Raw EEG acquisition
2. Noise removal
3. Signal cleaning
4. Sliding window segmentation
5. Subject-wise train-test split
6. Model input preparation
7. TCN training

---

# 🏋️ Training

| Hyperparameter | Value                |
| -------------- | -------------------- |
| Optimizer      | Adam                 |
| Batch Size     | 32                   |
| Loss Function  | Binary Cross Entropy |
| Architecture   | TCN                  |
| Output Layer   | Sigmoid              |
| Classification | Binary               |

### Regularization Techniques

* Batch Normalization
* Dropout
* Residual Learning

---

# 📈 Evaluation

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

Generated outputs include:

* Confusion Matrix
* ROC Curve
* Accuracy Curves
* Classification Metrics

---

# 🛠️ Tech Stack

| Category                | Technology       |
| ----------------------- | ---------------- |
| Programming Language    | Python           |
| Deep Learning           | PyTorch          |
| Data Processing         | NumPy, Pandas    |
| Visualization           | Matplotlib       |
| Machine Learning        | Scikit-learn     |
| Development Environment | Jupyter Notebook |

---

# 🔭 Future Work

* Cross-subject generalization
* Real-time lie detection system
* Transformer-based EEG architectures
* Attention-enhanced TCN models
* Portable EEG device integration
* Forensic decision-support systems

---

# 📚 References

1. Bai, S., Kolter, J.Z., Koltun, V. Temporal Convolutional Networks for Sequence Modeling.
2. Mahmoud, H.A.H. Deep Learning Model for EEG-Based Lie Detection.
3. Farah, N. et al. Neurophysiological Approaches to Lie Detection.
4. Wen, X., Li, W. Time Series Prediction Using LSTM Architectures.
5. LieWaves Dataset – Mendeley Data.

---

# 👩‍💻 Authors

**Adepu Tejaswini**
📧 [adeputejaswini7@gmail.com](mailto:adeputejaswini7@gmail.com)


---


