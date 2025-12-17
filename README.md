# Explainable Image Classification using Grad-CAM

This project implements an **Explainable Image Classification System** using a pre-trained **ResNet-50 Convolutional Neural Network** and **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize and interpret model predictions.

An interactive **Streamlit** web application allows users to upload images, view predicted classes, and understand *where the model focuses* while making decisions.

---

## 🚀 Features

- Image classification using **ResNet-50 CNN**
- **Grad-CAM heatmap visualization** for explainability
- Interactive **Streamlit-based UI**
- Uses **pre-trained ImageNet weights**
- Highlights regions influencing model predictions
- Demonstrates principles of **Explainable AI (XAI)**

---

## 🧠 Motivation

Deep learning models often act as *black boxes*, providing predictions without explanations.  
This project addresses that limitation by integrating **Grad-CAM**, enabling visual interpretation of CNN decisions and improving transparency and trust in AI systems.

---

## 🏗️ System Architecture

User
↓
Streamlit Web Interface
↓
Image Preprocessing
↓
ResNet-50 CNN
↓
Prediction
↓
Grad-CAM Heatmap Visualization

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Deep Learning Framework:** PyTorch  
- **Model:** ResNet-50 (Pre-trained)  
- **Explainability:** Grad-CAM  
- **Web Framework:** Streamlit  
- **Libraries:** NumPy, Pillow, OpenCV, Matplotlib  

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/explainable-image-classification-gradcam.git
cd explainable-image-classification-gradcam
