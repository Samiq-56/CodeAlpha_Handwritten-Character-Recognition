# ============================================
# PART 2: EMNIST Letter Recognition (A-Z)
# CodeAlpha ML Internship — Task 3
# ============================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import os

print("="*50)
print("EMNIST Letter Recognition (A-Z)")
print("="*50)

# ── Load Dataset ─────────────────────────────
print("Loading EMNIST dataset...")
ds_train, ds_test = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    as_supervised=True
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = label - 1  # 1-26 → 0-25
    return image, label

X_train_e, y_train_e = [], []
for img, lbl in ds_train.map(preprocess):
    X_train_e.append(img.numpy())
    y_train_e.append(lbl.numpy())

X_test_e, y_test_e = [], []
for img, lbl in ds_test.map(preprocess):
    X_test_e.append(img.numpy())
    y_test_e.append(lbl.numpy())

X_train_e = np.array(X_train_e)
y_train_e  = np.array(y_train_e)
X_test_e   = np.array(X_test_e)
y_test_e   = np.array(y_test_e)

print(f"Train: {X_train_e.shape[0]} | Test: {X_test_e.shape[0]}")

y_train_cat = to_categorical(y_train_e, 26)
y_test_cat  = to_categorical(y_test_e, 26)

class_names = [chr(65+i) for i in range(26)]

# ── Sample Images ─────────────────────────────
plt.figure(figsize=(13, 2))
for i in range(13):
    plt.subplot(1, 13, i+1)
    plt.imshow(X_train_e[i].reshape(28, 28), cmap='gray')
    plt.title(class_names[y_train_e[i]])
    plt.axis('off')
plt.suptitle("EMNIST Sample Images (A-Z)")
plt.tight_layout()
plt.savefig("emnist_samples.png")
plt.show()
print("Sample images saved!")

# ── Build Model ───────────────────────────────
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
    Dense(26, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ── Train ─────────────────────────────────────
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nTraining started...")
history = model.fit(
    X_train_e, y_train_cat,
    epochs=15, batch_size=128,
    validation_split=0.1,
    callbacks=[early_stop], verbose=1
)

# ── Evaluate ──────────────────────────────────
loss, acc = model.evaluate(X_test_e, y_test_cat, verbose=0)
print(f"\nEMNIST Test Accuracy: {acc*100:.2f}%")

y_pred = np.argmax(model.predict(X_test_e), axis=1)
print("\nClassification Report:")
print(classification_report(y_test_e, y_pred, target_names=class_names))

# ── Confusion Matrix ──────────────────────────
cm = confusion_matrix(y_test_e, y_pred)
plt.figure(figsize=(16,14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.title('EMNIST Confusion Matrix (A-Z)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("emnist_confusion_matrix.png")
plt.show()

# ── Training Curves ───────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Val')
ax1.set_title('EMNIST Accuracy')
ax1.legend()
ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Val')
ax2.set_title('EMNIST Loss')
ax2.legend()
plt.tight_layout()
plt.savefig("emnist_training_curves.png")
plt.show()

# ── Save Model ────────────────────────────────
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/emnist_model.keras")
print("\nEMNIST model saved!")
print("Task 3 COMPLETE!")