# ✋ Handwritten Character Recognition — CodeAlpha ML Internship

CNN-based model that recognizes handwritten digits (0–9) and letters (A–Z) using MNIST and EMNIST datasets.

---

## 📌 Task Info
- **Internship:** CodeAlpha Machine Learning
- **Task:** Task 3 — Handwritten Character Recognition
- **Datasets:** MNIST + EMNIST Letters

---

## 🧠 Model Architecture (CNN)

| Layer | Details |
|-------|---------|
| Conv2D + BatchNorm | 32 filters, 3×3 |
| Conv2D + MaxPooling | 32 filters |
| Conv2D + BatchNorm | 64 filters, 3×3 |
| Conv2D + MaxPooling | 64 filters |
| Dense + Dropout | 256 units |
| Output (Softmax) | 10 / 26 classes |

---

## 📊 Results

| Dataset | Classes | Accuracy |
|---------|---------|----------|
| MNIST (Digits) | 0–9 | **99.39%** |
| EMNIST (Letters) | A–Z | **93.64%** |

---

## 🚀 How to Run

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn pillow tensorflow-datasets

# Train digit recognition (MNIST)
python train_mnist.py

# Train letter recognition (EMNIST)
python train_emnist.py
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `train_mnist.py` | MNIST digit model training |
| `train_emnist.py` | EMNIST letter model training |
| `predict.py` | Predict on custom image |

---

## 🛠 Tech Stack
Python • TensorFlow • Keras • Scikit-learn • Matplotlib • TensorFlow Datasets

---

*CodeAlpha ML Internship — Task 3*
