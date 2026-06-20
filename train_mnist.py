# ============================================
# Task 3: Handwritten Character Recognition
# MNIST (Digits) + EMNIST (Letters)
# CodeAlpha ML Internship
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from emnist import extract_training_samples, extract_test_samples
from tensorflow.keras.datasets import mnist
import os

# ══════════════════════════════════════════
# PART 1: MNIST — Digit Recognition (0-9)
# ══════════════════════════════════════════
print("\n" + "="*50)
print("PART 1: MNIST Digit Recognition")
print("="*50)

(X_train_m, y_train_m), (X_test_m, y_test_m) = mnist.load_data()

X_train_m = X_train_m.astype('float32') / 255.0
X_test_m  = X_test_m.astype('float32')  / 255.0
X_train_m = X_train_m.reshape(-1, 28, 28, 1)
X_test_m  = X_test_m.reshape(-1, 28, 28, 1)

y_train_m_cat = to_categorical(y_train_m, 10)
y_test_m_cat  = to_categorical(y_test_m, 10)

# Sample images
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(X_train_m[i].reshape(28, 28), cmap='gray')
    plt.title(str(y_train_m[i]))
    plt.axis('off')
plt.suptitle("MNIST Sample Images (Digits)")
plt.tight_layout()
plt.savefig("mnist_samples.png")
plt.show()

# Build MNIST model
def build_cnn(num_classes):
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

mnist_model = build_cnn(10)
mnist_model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nTraining MNIST model...")
history_m = mnist_model.fit(
    X_train_m, y_train_m_cat,
    epochs=15, batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop], verbose=1
)

loss_m, acc_m = mnist_model.evaluate(X_test_m, y_test_m_cat, verbose=0)
print(f"\nMNIST Test Accuracy: {acc_m*100:.2f}%")

y_pred_m = np.argmax(mnist_model.predict(X_test_m), axis=1)
print("\nMNIST Classification Report:")
print(classification_report(y_test_m, y_pred_m,
      target_names=[str(i) for i in range(10)]))

# Confusion matrix
cm_m = confusion_matrix(y_test_m, y_pred_m)
plt.figure(figsize=(10,8))
sns.heatmap(cm_m, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title('MNIST Confusion Matrix (Digits)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("mnist_confusion_matrix.png")
plt.show()

# Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history_m.history['accuracy'], label='Train')
ax1.plot(history_m.history['val_accuracy'], label='Val')
ax1.set_title('MNIST Accuracy')
ax1.legend()
ax2.plot(history_m.history['loss'], label='Train')
ax2.plot(history_m.history['val_loss'], label='Val')
ax2.set_title('MNIST Loss')
ax2.legend()
plt.tight_layout()
plt.savefig("mnist_training_curves.png")
plt.show()

os.makedirs("saved_model", exist_ok=True)
mnist_model.save("saved_model/mnist_model.keras")
print("MNIST model saved!")

# ══════════════════════════════════════════
# PART 2: EMNIST — Letter Recognition (A-Z)
# ══════════════════════════════════════════
print("\n" + "="*50)
print("PART 2: EMNIST Letter Recognition (A-Z)")
print("="*50)

import tensorflow_datasets as tfds

print("Loading EMNIST letters dataset...")
ds_train, ds_test = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = label - 1  # 1-26 → 0-25
    return image, label

import tensorflow as tf

X_train_e, y_train_e = [], []
for img, lbl in ds_train.map(preprocess):
    X_train_e.append(img.numpy())
    y_train_e.append(lbl.numpy())

X_test_e, y_test_e = [], []
for img, lbl in ds_test.map(preprocess):
    X_test_e.append(img.numpy())
    y_test_e.append(lbl.numpy())

X_train_e = np.array(X_train_e)
y_train_e = np.array(y_train_e)
X_test_e  = np.array(X_test_e)
y_test_e  = np.array(y_test_e)

print(f"EMNIST Train: {X_train_e.shape[0]} | Test: {X_test_e.shape[0]}")

y_train_e_cat = to_categorical(y_train_e, 26)
y_test_e_cat  = to_categorical(y_test_e, 26)

class_names = [chr(65+i) for i in range(26)]

# Sample images
plt.figure(figsize=(13, 2))
for i in range(13):
    plt.subplot(1, 13, i+1)
    plt.imshow(X_train_e[i].reshape(28, 28), cmap='gray')
    plt.title(class_names[y_train_e[i]])
    plt.axis('off')
plt.suptitle("EMNIST Sample Images (Letters A-Z)")
plt.tight_layout()
plt.savefig("emnist_samples.png")
plt.show()

emnist_model = build_cnn(26)
emnist_model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nTraining EMNIST model...")
history_e = emnist_model.fit(
    X_train_e, y_train_e_cat,
    epochs=15, batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop], verbose=1
)

loss_e, acc_e = emnist_model.evaluate(X_test_e, y_test_e_cat, verbose=0)
print(f"\nEMNIST Test Accuracy: {acc_e*100:.2f}%")

y_pred_e = np.argmax(emnist_model.predict(X_test_e), axis=1)
print("\nEMNIST Classification Report:")
print(classification_report(y_test_e, y_pred_e, target_names=class_names))

# Confusion Matrix
cm_e = confusion_matrix(y_test_e, y_pred_e)
plt.figure(figsize=(16,14))
sns.heatmap(cm_e, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.title('EMNIST Confusion Matrix (Letters A-Z)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("emnist_confusion_matrix.png")
plt.show()

# Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history_e.history['accuracy'], label='Train')
ax1.plot(history_e.history['val_accuracy'], label='Val')
ax1.set_title('EMNIST Accuracy')
ax1.legend()
ax2.plot(history_e.history['loss'], label='Train')
ax2.plot(history_e.history['val_loss'], label='Val')
ax2.set_title('EMNIST Loss')
ax2.legend()
plt.tight_layout()
plt.savefig("emnist_training_curves.png")
plt.show()

emnist_model.save("saved_model/emnist_model.keras")
print("EMNIST model saved!")

print("\n" + "="*50)
print("FINAL RESULTS SUMMARY")
print("="*50)
print(f"MNIST  (Digits 0-9):   {acc_m*100:.2f}% accuracy")
print(f"EMNIST (Letters A-Z):  {acc_e*100:.2f}% accuracy")
print("="*50)
print("All done! Task 3 Complete!")