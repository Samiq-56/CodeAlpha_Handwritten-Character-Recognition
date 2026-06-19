# ✋ Handwritten Digit Recognition — CodeAlpha ML Internship

A CNN-based model that recognizes handwritten digits (0–9) using the MNIST dataset with **99% accuracy**.

---

## 📌 Task Info
- **Internship:** CodeAlpha Machine Learning
- **Task:** Task 3 — Handwritten Character Recognition
- **Dataset:** MNIST (60,000 train / 10,000 test images)

---

## 🧠 Model Architecture (CNN)

| Layer | Details |
|-------|---------|
| Conv2D + BatchNorm | 32 filters, 3×3 |
| Conv2D + MaxPooling | 32 filters |
| Conv2D + BatchNorm | 64 filters, 3×3 |
| Conv2D + MaxPooling | 64 filters |
| Dense + Dropout | 256 units |
| Output (Softmax) | 10 classes |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **99%** |
| Precision | 0.99 |
| Recall | 0.99 |
| F1-Score | 0.99 |

---

## 🚀 How to Run

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn pillow
python train_model.py
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `train_model.py` | Model training |
| `predict.py` | Predict on custom image |
| `saved_model/` | Saved trained model |

---

## 🛠 Tech Stack
Python • TensorFlow • Keras • Scikit-learn • Matplotlib

---

*CodeAlpha ML Internship — Task 3*
