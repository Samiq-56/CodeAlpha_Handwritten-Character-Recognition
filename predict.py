# ============================================
# Predict on custom image
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import sys

# Load trained model
model = load_model("saved_model/mnist_cnn_model.h5")
print("Model loaded!")

def predict_digit(image_path):
    # Load & preprocess image
    img = Image.open(image_path).convert('L')   # Grayscale
    img = ImageOps.invert(img)                   # Invert colors
    img = img.resize((28, 28))                   # Resize to 28x28
    
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Show result
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {digit} | Confidence: {confidence:.1f}%")
    plt.axis('off')
    plt.show()
    
    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence:.1f}%")

# Usage: python predict.py your_image.png
if len(sys.argv) > 1:
    predict_digit(sys.argv[1])
else:
    print("Usage: python predict.py <image_path>")