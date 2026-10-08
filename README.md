# 📚 Literature Review: PCOS Detection Using Machine Learning

This repository presents a literature review project titled **"Polycystic Ovary Syndrome (PCOS) and Machine Learning"** authored by **Riya Garg**, **Sarthak Chaturvedi**, and **Simran Pahwa** under the Department of Computer Science, **Keshav Mahavidyalaya (University of Delhi)**.

## 🧠 Overview

**Polycystic Ovary Syndrome (PCOS)** is a complex endocrine disorder that affects millions of women of reproductive age. Traditional diagnostic methods are often time-consuming, resource-intensive, and suffer from inconsistencies. With the rapid advancement of **Machine Learning (ML)** and **Deep Learning (DL)**, there is great potential to improve the accuracy, speed, and accessibility of PCOS diagnosis.

This review analyzes various **ML/DL algorithms** used in PCOS detection, evaluates their performance, identifies gaps in current methodologies, and highlights the use of **Explainable AI (XAI)** for clinical applicability.

## 📌 Objectives

- To study and consolidate research on **PCOS prediction using machine learning**.
- To compare the performance of models like **Logistic Regression, SVM, Random Forest, ANN, CNN**, etc.
- To highlight the role of **feature selection** techniques (RFE, Chi-Squared, Mutual Information).
- To explore the integration of **explainable AI methods** (especially **SHAP**) for model interpretability.
- To discuss current **limitations**, challenges, and **future research directions** in this domain.

## 🛠️ Methodology

- Reviewed **35 peer-reviewed articles** from **Scopus, PubMed, IEEE Xplore, and Web of Science**.
- Applied strict **inclusion-exclusion criteria** (2020–2025, SJR ≥ 0.15, English, peer-reviewed).
- Analyzed studies with models trained on both **public** and **clinical datasets**.
- Conducted **comparative analysis** of accuracy, AUC scores, and model robustness.
- Evaluated use of **Explainable AI** like **SHAP values**, **LIME**, and **Q-lattice**.

## 📊 Key Findings

- ML models achieved **up to 100% accuracy** using optimized features and ensemble models.
- **Random Forest**, **Gaussian Naive Bayes**, and **Stacked Classifiers** showed consistently high performance.
- **SHAP** emerged as a leading explainability tool, aiding clinician trust in AI predictions.
- The biggest bottlenecks include **small sample sizes**, **data imbalance**, and **lack of diverse datasets**.
- Need for real-world validation and integration into **Electronic Health Record (EHR)** systems.

## 💡 Future Directions

- Create models trained on **multi-modal datasets** (genomic, imaging, clinical).
- Enhance fairness through **subgroup analysis** by age, ethnicity, and geography.
- Integrate lightweight and explainable ML models into **clinical workflows**.
- Ensure compliance with **ethical standards** and **regulatory guidelines** for medical AI.

## 📄 Paper Details

- **Title**: Polycystic Ovary Syndrome (PCOS) and Machine Learning  
- **Authors**: Riya Garg, Sarthak Chaturvedi, Simran Pahwa  
- **Institution**: Keshav Mahavidyalaya, University of Delhi  
- **Type**: Undergraduate Literature Review Project  
- **Status**: Not published (Academic submission)  

## 📂 Files in This Repository

- `PCOS_ML_Literature_Review.pdf` – Final version of the literature review paper.
  
## 🙌 Acknowledgments

We would like to thank our faculty members at **Keshav Mahavidyalaya** for their guidance and motivation throughout this research.

---

## Multi-Scale ConvNeXt with CBAM for Ovarian Tumor Classification

A deep learning model for **binary classification of ovarian histological image patches** into **tumour** and **non-tumour** classes using a pretrained **ConvNeXt-Tiny backbone enhanced with CBAM-based attention and multi-scale feature fusion**.

## Overview

This project implements a multi-scale deep learning architecture designed to classify ovarian histological image patches.

The model combines:

* **ConvNeXt-Tiny** as the pretrained backbone
* **CBAM (Convolutional Block Attention Module)** for channel and spatial attention
* **Multi-scale feature fusion** using the final two ConvNeXt stages
* **Weighted Cross-Entropy Loss** with label smoothing
* **Test-Time Augmentation (TTA)** for evaluation
* **Adaptive classification threshold selection**
* **Grad-CAM** for model interpretability

The complete implementation was developed and trained using **Google Colab with GPU acceleration**.

---

## Dataset

The dataset contains ovarian histological image patches divided into two classes:

```text
ovarian/
├── non-tumour/
└── tumour/
```

The notebook loads the dataset using `torchvision.datasets.ImageFolder`.

### Dataset Statistics

| Class      |     Images |
| ---------- | ---------: |
| Non-tumour |     13,400 |
| Tumour     |     19,710 |
| **Total**  | **33,110** |

The dataset is divided into:

* **80% training**
* **20% validation**

The validation set contains **6,622 images**.

---

## Image Preprocessing

Images are resized to:

```text
224 × 224
```

### Training Augmentation

The training pipeline includes:

* Random horizontal flip
* Random vertical flip
* Random rotation up to 20°
* Color jitter
* Image normalization

The validation pipeline uses resizing and normalization without random augmentation.

The normalization values correspond to the standard ImageNet normalization:

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

---

## Model Architecture

The proposed model is based on **ConvNeXt-Tiny** with additional attention and multi-scale feature processing.

### Architecture

```text
Input Image
     │
     ▼
ConvNeXt-Tiny Backbone
     │
     ├───────────────┐
     │               │
     ▼               ▼
 Stage 2          Stage 3
 384 channels     768 channels
     │               │
     ▼               ▼
   CBAM             CBAM
     │               │
     ▼               │
  1×1 Projection     │
     │               │
     └───────┬───────┘
             ▼
      Multi-Scale Fusion
             │
             ▼
    Global Average Pooling
             │
             ▼
          LayerNorm
             │
             ▼
          Dropout
             │
             ▼
       Fully Connected
             │
             ▼
     Tumour / Non-tumour
```

The implementation uses a pretrained ConvNeXt-Tiny model from `timm` and extracts multi-scale features from the final two stages.

---

## CBAM Attention

The model incorporates **Convolutional Block Attention Module (CBAM)** at two feature stages.

CBAM consists of:

### 1. Channel Attention

Channel attention uses both:

* Global Average Pooling
* Global Max Pooling

The resulting features are passed through fully connected layers and a sigmoid activation.

### 2. Spatial Attention

Spatial attention calculates:

* Channel-wise average
* Channel-wise maximum

These are concatenated and processed using a convolution layer.

The model applies CBAM independently to the 384-channel and 768-channel feature representations.

---

## Multi-Scale Feature Fusion

Features from two ConvNeXt stages are used:

* Stage 2: **384 channels**
* Stage 3: **768 channels**

The Stage 2 features are projected from 384 to 768 channels using a `1×1` convolution and then spatially resized to match Stage 3.

The two representations are fused through element-wise addition:

```python
fused = s2_up + s3
```

This allows the classifier to use information from multiple feature levels.

---

## Fine-Tuning Strategy

Initially, the model parameters are frozen.

The following components are then made trainable:

* Last ConvNeXt backbone stage
* CBAM for Stage 2
* CBAM for Stage 3
* Feature projection layer
* Classification head

The resulting model has approximately **15.86 million trainable parameters**.

---

## Loss Function

The model uses weighted Cross-Entropy Loss with:

```text
Label smoothing = 0.1
```

Class weights are calculated based on the number of samples in each class.

```python
criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=0.1
)
```

This was used instead of focal loss in the final implementation because the weighted Cross-Entropy configuration was found to be more stable for the task.

---

## Training

The model is trained using:

| Parameter       | Value             |
| --------------- | ----------------- |
| Optimizer       | AdamW             |
| Learning Rate   | 1e-4              |
| Weight Decay    | 0.05              |
| Scheduler       | CosineAnnealingLR |
| Epochs          | 12                |
| Batch Size      | 64                |
| Label Smoothing | 0.1               |
| GPU             | NVIDIA Tesla T4   |

## A checkpointing mechanism is implemented so that training can resume after interruption. The best-performing model and training history are also saved to Google Drive.

## Test-Time Augmentation

During evaluation, predictions are generated using four transformations:

1. Original image
2. Horizontal flip
3. Vertical flip
4. 90° rotation

The resulting probability predictions are averaged before classification.

This provides an ensemble-like prediction from multiple transformed versions of the same image.

---

## Results

The model was evaluated on **6,622 validation images**.

### Final Performance

| Metric         |     Result |
| -------------- | ---------: |
| Accuracy       | **98.97%** |
| F1-Score       | **99.24%** |
| ROC-AUC        | **99.82%** |
| Best Threshold |   **0.70** |

The adaptive threshold was searched between 0.20 and 0.80, with the best F1-score obtained at a threshold of **0.70**.

### Classification Report

| Class      | Precision | Recall | F1-Score |
| ---------- | --------: | -----: | -------: |
| Non-tumour |      0.99 |   0.99 |     0.99 |
| Tumour     |      0.99 |   0.99 |     0.99 |

The validation set contained 2,680 non-tumour and 3,942 tumour samples.

### Confusion Matrix

```text
                 Predicted
               Non-tumour  Tumour
Actual
Non-tumour        2654       26
Tumour              34     3908
```

The model produced **60 incorrect predictions** on the 6,622-image validation set.

---

## Model Interpretability

**Grad-CAM (Gradient-weighted Class Activation Mapping)** is used to visualize the image regions contributing to the model's predictions.

The implementation generates activation maps from the spatial attention convolution layer of the Stage 3 CBAM module and overlays the resulting heatmap on the original image.

This provides a visual indication of the regions influencing the tumour classification decision.

---

## Technologies Used

* Python
* PyTorch
* Torchvision
* `timm`
* NumPy
* OpenCV
* Scikit-learn
* Matplotlib
* Google Colab
* CUDA / NVIDIA GPU

---

## Installation

Install the required libraries using:

```bash
pip install torch torchvision timm scikit-learn opencv-python
```

The notebook imports the required deep learning, image processing, visualization, and evaluation libraries.

---

## Running the Project

### 1. Prepare the Dataset

Place the dataset in the following structure:

```text
ovarian/
├── non-tumour/
└── tumour/
```

### 2. Open the Notebook

Open the notebook in **Google Colab** and enable GPU acceleration.

### 3. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Load the Dataset

The notebook expects the dataset archive at:

```text
/content/drive/MyDrive/Datasets/ovarian.zip
```

It extracts the dataset into:

```text
/content/ovarian
```

### 5. Train the Model

Run the notebook cells sequentially to:

* Load the dataset
* Apply preprocessing
* Build the ConvNeXt + CBAM model
* Fine-tune the model
* Train for 12 epochs
* Save checkpoints
* Evaluate the model

### 6. Evaluate

The evaluation section calculates:

* Classification report
* Confusion matrix
* F1-score
* ROC-AUC
* Optimal classification threshold

### 7. Generate Grad-CAM

The final section generates Grad-CAM visualizations to inspect the model's attention regions.

---

## Checkpoints and Training History

The notebook saves:

```text
convnext_checkpoint.pth
best_model.pth
training_results.json
```

The checkpoint contains:

* Model state
* Optimizer state
* Scheduler state
* Best validation accuracy
* Training history

This allows interrupted training to resume from the saved checkpoint.

---

## Project Structure

A recommended repository structure is:

```text
ConvNeXt-OvarianTumor/
│
├── ConvNeXt_OvarianTumor.ipynb
├── README.md
│
├── models/
│   └── best_model.pth
│
├── results/
│   └── training_results.json
│
└── visualizations/
    └── gradcam/
```

The dataset itself should not be uploaded to the repository if it is large or subject to separate dataset licensing/usage conditions.

---

## Key Features

* Pretrained ConvNeXt-Tiny backbone
* Multi-scale feature extraction
* Dual CBAM attention modules
* Feature projection and fusion
* Weighted Cross-Entropy loss
* Label smoothing
* Adaptive classification threshold
* Test-Time Augmentation
* Checkpoint-based training recovery
* Grad-CAM interpretability
* GPU acceleration

---

## Limitations

The reported results are based on the validation split used in the notebook. The project does not, in the provided implementation, include an independent external test-set evaluation.

Therefore, the reported performance should be interpreted as **validation performance**, rather than evidence of clinical deployment or clinical diagnostic performance.

---

## Disclaimer

This project is intended for **research and educational purposes**. It is not a clinical diagnostic system and should not be used as a substitute for professional pathological assessment.

---

## Author

**Riya Garg**

B.Sc. (Hons.) Computer Science
University of Delhi

---

## Acknowledgements

This project uses open-source deep learning libraries including PyTorch, Torchvision, `timm`, Scikit-learn, OpenCV, and Matplotlib.


**Feel free to fork or use parts of this project for your own research or academic work.**  
If you find this helpful, don't forget to ⭐ the repository!

