# ============================================
# AirType — Live Digit Recognition Demo
# Draw with mouse → Model predicts real-time
# CodeAlpha ML Internship — Task 3
# ============================================

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ── Load Model ────────────────────────────────
print("Loading model...")
model = load_model("saved_model/mnist_model.keras")
print("Model loaded!")

# ── Canvas Setup ──────────────────────────────
CANVAS_W, CANVAS_H = 800, 600
DRAW_W,   DRAW_H   = 280, 280  # drawing area (top-left)
canvas    = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
drawing   = False
last_x, last_y = -1, -1
prediction, confidence = -1, 0.0

def preprocess_canvas(draw_area):
    """Resize 280x280 → 28x28 → normalize → predict"""
    gray    = cv2.cvtColor(draw_area, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    norm    = resized.astype('float32') / 255.0
    inp     = norm.reshape(1, 28, 28, 1)
    return inp

def draw_ui():
    global canvas
    # Sirf right panel clear karo, drawing area mat chhuo
    canvas[:, DRAW_W:] = 0

    # Drawing area border
    cv2.rectangle(canvas, (0, 0), (DRAW_W, DRAW_H),
                  (80, 80, 80), 2)
    cv2.putText(canvas, "Draw Here", (85, DRAW_H + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150,150,150), 1)

    # Right panel
    panel_x = DRAW_W + 40

    # Title
    cv2.putText(canvas, "MNIST Live Demo",
                (panel_x, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

    cv2.putText(canvas, "CodeAlpha | Task 3",
                (panel_x, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

    # Divider
    cv2.line(canvas, (panel_x, 95), (CANVAS_W-20, 95),
             (60,60,60), 1)

    # Prediction
    cv2.putText(canvas, "Prediction:",
                (panel_x, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)

    if prediction >= 0:
        cv2.putText(canvas, str(prediction),
                    (panel_x + 10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 255, 100), 10)

        cv2.putText(canvas, "Confidence:",
                    (panel_x, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)

        # Confidence bar
        bar_w = int((confidence / 100) * 250)
        cv2.rectangle(canvas, (panel_x, 285),
                      (panel_x + 250, 310), (50,50,50), -1)
        color = (0,255,100) if confidence > 80 else \
                (0,200,255) if confidence > 60 else (0,100,255)
        cv2.rectangle(canvas, (panel_x, 285),
                      (panel_x + bar_w, 310), color, -1)
        cv2.putText(canvas, f"{confidence:.1f}%",
                    (panel_x + bar_w + 5, 307),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    else:
        cv2.putText(canvas, "---",
                    (panel_x + 10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (80,80,80), 5)

    # Controls
    cv2.line(canvas, (panel_x, 340), (CANVAS_W-20, 340),
             (60,60,60), 1)
    cv2.putText(canvas, "Controls:",
                (panel_x, 365),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.putText(canvas, "C  — Clear canvas",
                (panel_x, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)
    cv2.putText(canvas, "ESC — Quit",
                (panel_x, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150,150,150), 1)

def mouse_draw(event, x, y, flags, param):
    global drawing, last_x, last_y, prediction, confidence

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_x, last_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if 0 <= x <= DRAW_W and 0 <= y <= DRAW_H:
            cv2.line(canvas, (last_x, last_y), (x, y),
                     (255, 255, 255), 18)
            last_x, last_y = x, y

            # Real-time prediction
            draw_area = canvas[0:DRAW_H, 0:DRAW_W]
            inp = preprocess_canvas(draw_area)
            preds = model.predict(inp, verbose=0)[0]
            prediction = int(np.argmax(preds))
            confidence = float(np.max(preds)) * 100

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_x, last_y = -1, -1

# ── Main Loop ─────────────────────────────────
cv2.namedWindow("MNIST Live Demo")
cv2.setMouseCallback("MNIST Live Demo", mouse_draw)

print("\nLive Demo started!")
print("Draw a digit (0-9) with mouse in the black area.")
print("C = Clear | ESC = Quit")

while True:
    draw_ui()

    # Paste drawn pixels back
    drawn = canvas[0:DRAW_H, 0:DRAW_W].copy()

    cv2.imshow("MNIST Live Demo", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:      # ESC
        break
    elif key == ord('c') or key == ord('C'):
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        prediction, confidence = -1, 0.0
        print("Canvas cleared!")

cv2.destroyAllWindows()
print("Demo closed!")