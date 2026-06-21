# ============================================================
# ✍️  DigiSense AI — Handwritten Digit Recognition
# Author  : Syed Samiq Abbas Bukhari
# Company : CodeAlpha
# Project : ML Internship — Task 3
# Model   : CNN trained on MNIST | Accuracy: 99.39%
# ============================================================

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="DigiSense AI | CodeAlpha",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #020817 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1526 100%) !important;
    border-right: 1px solid rgba(139,92,246,0.14) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Hero ── */
.hero-wrap {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #0e0a1e 100%);
    border: 1px solid rgba(139,92,246,0.20);
    border-radius: 20px;
    padding: 36px 42px 30px;
    margin-bottom: 26px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content:'';
    position:absolute; top:-100px; right:-100px;
    width:360px; height:360px;
    background: radial-gradient(circle, rgba(139,92,246,0.13) 0%, transparent 70%);
    pointer-events:none;
}
.hero-wrap::after {
    content:'';
    position:absolute; bottom:-80px; left:-80px;
    width:280px; height:280px;
    background: radial-gradient(circle, rgba(56,189,248,0.07) 0%, transparent 70%);
    pointer-events:none;
}
.hero-eyebrow {
    display:inline-flex; align-items:center; gap:8px;
    background: rgba(139,92,246,0.12);
    border: 1px solid rgba(139,92,246,0.30);
    color: #a78bfa !important;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 11px; font-weight: 600;
    letter-spacing: 1.4px; text-transform: uppercase;
    margin-bottom: 14px;
}
.hero-title {
    font-size: clamp(28px, 4.5vw, 48px);
    font-weight: 900; line-height: 1.1;
    margin: 0 0 12px;
    background: linear-gradient(135deg, #f1f5f9 0%, #a78bfa 45%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 14px; color: #64748b !important;
    font-weight: 400; margin: 0; line-height: 1.6;
}
.hero-author {
    margin-top: 22px;
    display: flex; align-items: center; gap: 12px;
}
.av {
    width: 40px; height: 40px; border-radius: 50%;
    background: linear-gradient(135deg, #8b5cf6, #38bdf8);
    display:flex; align-items:center; justify-content:center;
    font-size:15px; font-weight:800; color:#fff !important;
    flex-shrink:0;
}
.av-name { font-size:13px; font-weight:700; color:#e2e8f0 !important; }
.av-role { font-size:11px; color:#64748b !important; }

/* ── Stat Pills ── */
.stat-row {
    display:flex; gap:12px; flex-wrap:wrap;
    margin-bottom:24px;
}
.stat-pill {
    background: #0d1526;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 20px;
    flex: 1; min-width: 120px;
    position: relative; overflow: hidden;
}
.stat-pill-top {
    position:absolute; top:0; left:0; right:0; height:2px;
    border-radius:10px 10px 0 0;
}
.stat-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px; font-weight: 800;
    line-height: 1; margin: 6px 0 3px;
}
.stat-lbl { font-size: 10px; color:#64748b !important; font-weight:600;
             text-transform:uppercase; letter-spacing:0.8px; }

/* ── Canvas Panel ── */
.canvas-panel {
    background: #0d1526;
    border: 1px solid rgba(139,92,246,0.18);
    border-radius: 16px;
    padding: 24px;
}
.canvas-title {
    font-size:13px; font-weight:700; color:#94a3b8 !important;
    text-transform:uppercase; letter-spacing:0.8px;
    margin-bottom:14px;
    display:flex; align-items:center; gap:8px;
}

/* ── Result Panel ── */
.result-panel {
    background: #0d1526;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px;
    height: 100%;
}

/* ── Digit Display ── */
.digit-display {
    border-radius: 14px;
    padding: 24px 20px 20px;
    text-align: center;
    margin-bottom: 14px;
    position: relative; overflow:hidden;
}
.digit-display-high {
    background: linear-gradient(135deg, #0f0520 0%, #0a0a1f 100%);
    border: 2px solid #8b5cf6;
    box-shadow: 0 0 32px rgba(139,92,246,0.18);
}
.digit-display-low {
    background: #0a0f1e;
    border: 1.5px solid rgba(255,255,255,0.08);
}
.digit-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 96px; font-weight: 800;
    line-height: 1; margin: 0;
    background: linear-gradient(135deg, #c4b5fd, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.conf-label {
    font-size: 12px; color: #64748b !important;
    text-transform: uppercase; letter-spacing: 0.8px;
    margin: 10px 0 4px;
}
.conf-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px; font-weight: 800;
    color: #4ade80 !important;
}

/* ── Prob Bars ── */
.prob-row {
    display: flex; align-items:center; gap:10px;
    margin-bottom: 6px;
}
.prob-digit-label {
    font-family:'JetBrains Mono',monospace;
    font-size:13px; font-weight:700;
    color:#94a3b8 !important;
    width:16px; text-align:right; flex-shrink:0;
}
.prob-bar-track {
    flex:1; background:#1e293b; border-radius:6px; height:8px; overflow:hidden;
}
.prob-bar-fill {
    height:100%; border-radius:6px;
    transition: width 0.5s ease;
}
.prob-pct {
    font-family:'JetBrains Mono',monospace;
    font-size:11px; color:#64748b !important;
    width:40px; text-align:right; flex-shrink:0;
}

/* ── Empty State ── */
.empty-state {
    text-align:center; padding:40px 20px;
    color:#334155 !important;
}
.empty-icon { font-size:48px; margin-bottom:12px; }
.empty-text { font-size:13px; }

/* ── Section Header ── */
.sec-head {
    display:flex; align-items:center; gap:10px;
    margin: 28px 0 16px;
}
.sec-icon {
    width:32px; height:32px; border-radius:8px;
    background: rgba(139,92,246,0.12);
    border:1px solid rgba(139,92,246,0.22);
    display:flex; align-items:center; justify-content:center;
    font-size:14px; flex-shrink:0;
}
.sec-title { font-size:15px; font-weight:700; color:#f1f5f9 !important; margin:0; }
.sec-sub   { font-size:11px; color:#475569 !important; margin:0; }

/* ── Info Cards row ── */
.info-grid {
    display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr));
    gap:12px; margin-top:8px;
}
.info-card {
    background:#0d1526;
    border:1px solid rgba(255,255,255,0.06);
    border-radius:12px; padding:16px;
}
.info-card-icon { font-size:20px; margin-bottom:8px; }
.info-card-title { font-size:11px; font-weight:700; color:#94a3b8 !important;
                   text-transform:uppercase; letter-spacing:0.6px; }
.info-card-text  { font-size:12px; color:#64748b !important; margin-top:4px; line-height:1.5; }

/* ── Footer ── */
.footer {
    text-align:center; padding:28px 0 12px;
    border-top:1px solid rgba(255,255,255,0.06);
    margin-top:48px;
}
.footer-brand { font-size:13px; font-weight:800; color:#8b5cf6 !important; }
.footer-sub   { font-size:11px; color:#334155 !important; margin-top:4px; }

/* ── Streamlit overrides ── */
hr[data-testid="stDivider"] { border-color:rgba(255,255,255,0.06)!important; }
[data-testid="stButton"]>button {
    background: linear-gradient(135deg,#7c3aed,#6366f1) !important;
    border:none!important; border-radius:10px!important;
    color:#fff!important; font-weight:700!important;
    font-size:13px!important;
    box-shadow:0 4px 20px rgba(124,58,237,0.25)!important;
}
[data-testid="stButton"]>button:hover {
    transform:translateY(-1px)!important;
    box-shadow:0 6px 28px rgba(124,58,237,0.35)!important;
}
::-webkit-scrollbar { width:5px; background:#0a0f1e; }
::-webkit-scrollbar-thumb { background:#1e293b; border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
@st.cache_resource
def load_mnist_model():
    return load_model("saved_model/mnist_model.keras")

model = load_mnist_model()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0 10px'>
      <div style='font-size:38px'>✍️</div>
      <div style='font-size:16px;font-weight:800;color:#a78bfa;margin-top:6px'>DigiSense AI</div>
      <div style='font-size:10px;color:#475569;letter-spacing:1.2px;text-transform:uppercase;margin-top:3px'>by CodeAlpha</div>
    </div>
    <hr style='border-color:rgba(139,92,246,0.15);margin:12px 0'>
    """, unsafe_allow_html=True)

    st.markdown("**🧠 Model Info**")
    st.markdown("""
    <div style='font-size:12px;color:#64748b;line-height:2.1'>
    📐 Architecture: CNN<br>
    📦 Dataset: MNIST (70,000 samples)<br>
    🎯 Test Accuracy: 99.39%<br>
    🔢 Output Classes: 10 (0–9)<br>
    📏 Input Shape: 28 × 28 × 1
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("**📋 How to Use**")
    st.markdown("""
    <div style='font-size:12px;color:#64748b;line-height:2.1'>
    1. Draw a digit (0–9) on the canvas<br>
    2. Prediction appears instantly<br>
    3. Click Clear to draw again<br>
    4. Try different writing styles!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#334155;text-align:center;line-height:1.8'>
      <div style='color:#a78bfa;font-weight:800;font-size:13px'>Syed Samiq Abbas Bukhari</div>
      ML Intern · CodeAlpha<br>
      Task 3 — Digit Recognition
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────
# HERO
# ─────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
  <div class='hero-eyebrow'>✍️ &nbsp; Computer Vision · Deep Learning</div>
  <h1 class='hero-title'>Handwritten Digit<br>Recognition</h1>
  <p class='hero-sub'>
    Draw any digit from 0 to 9 on the canvas — a CNN trained on 70,000 MNIST samples<br>
    recognizes your handwriting in real-time with 99.39% accuracy.
  </p>
  <div class='hero-author'>
    <div class='av'>S</div>
    <div>
      <div class='av-name'>Syed Samiq Abbas Bukhari</div>
      <div class='av-role'>ML Intern · CodeAlpha · Task 3</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Stat Pills ────────────────────────────
st.markdown("""
<div class='stat-row'>
  <div class='stat-pill'>
    <div class='stat-pill-top' style='background:linear-gradient(90deg,#8b5cf6,#6366f1)'></div>
    <div class='stat-lbl'>Accuracy</div>
    <div class='stat-val' style='color:#a78bfa'>99.39%</div>
  </div>
  <div class='stat-pill'>
    <div class='stat-pill-top' style='background:linear-gradient(90deg,#38bdf8,#0ea5e9)'></div>
    <div class='stat-lbl'>Architecture</div>
    <div class='stat-val' style='color:#38bdf8;font-size:16px'>CNN</div>
  </div>
  <div class='stat-pill'>
    <div class='stat-pill-top' style='background:linear-gradient(90deg,#4ade80,#22c55e)'></div>
    <div class='stat-lbl'>Training Samples</div>
    <div class='stat-val' style='color:#4ade80'>70K</div>
  </div>
  <div class='stat-pill'>
    <div class='stat-pill-top' style='background:linear-gradient(90deg,#f59e0b,#f97316)'></div>
    <div class='stat-lbl'>Output Classes</div>
    <div class='stat-val' style='color:#fbbf24'>10</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# MAIN LAYOUT — Canvas + Result
# ─────────────────────────────────────────
left, right = st.columns([1.15, 1], gap="large")

with left:
    st.markdown("""
    <div class='canvas-title'>
      <span style='font-size:16px'>🖊️</span> Draw Your Digit
    </div>
    """, unsafe_allow_html=True)

    canvas = st_canvas(
        fill_color="black",
        stroke_width=20,
        stroke_color="white",
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    st.caption("Draw clearly inside the box · Use thick strokes for best results")

with right:
    st.markdown("<div class='result-panel'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='canvas-title'>
      <span style='font-size:16px'>🤖</span> AI Prediction
    </div>
    """, unsafe_allow_html=True)

    has_drawing = (
        canvas.image_data is not None and
        canvas.image_data.astype(np.uint8).sum() > 0
    )

    if has_drawing:
        img = canvas.image_data.astype(np.uint8)
        img_pil = Image.fromarray(img).convert("L")
        img_resized = img_pil.resize((28, 28))
        img_array = np.array(img_resized).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        preds = model.predict(img_array, verbose=0)[0]
        digit = int(np.argmax(preds))
        conf  = float(np.max(preds)) * 100

        high_conf = conf >= 70

        # Digit box
        st.markdown(f"""
        <div class='digit-display {"digit-display-high" if high_conf else "digit-display-low"}'>
          <div class='digit-number'>{digit}</div>
          <div class='conf-label'>Confidence</div>
          <div class='conf-value'>{conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence badge
        if conf >= 90:
            badge, badge_color = "🟢 Very High Confidence", "#4ade80"
        elif conf >= 70:
            badge, badge_color = "🟡 High Confidence", "#fbbf24"
        elif conf >= 50:
            badge, badge_color = "🟠 Moderate Confidence", "#f97316"
        else:
            badge, badge_color = "🔴 Low Confidence — try redrawing", "#f87171"

        st.markdown(f"<div style='font-size:12px;color:{badge_color};font-weight:600;margin-bottom:14px'>{badge}</div>",
                    unsafe_allow_html=True)

        # Probability bars for all digits
        st.markdown("<div style='font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;margin-bottom:10px'>All Digit Probabilities</div>",
                    unsafe_allow_html=True)

        for i, p in enumerate(preds):
            pct = p * 100
            is_top = (i == digit)
            bar_color = "linear-gradient(90deg,#8b5cf6,#6366f1)" if is_top else "#1e293b"
            fill_color = "#8b5cf6" if is_top else "#334155"
            label_color = "#c4b5fd" if is_top else "#475569"

            st.markdown(f"""
            <div class='prob-row'>
              <div class='prob-digit-label' style='color:{label_color}'>{i}</div>
              <div class='prob-bar-track'>
                <div class='prob-bar-fill' style='width:{pct:.1f}%;background:{fill_color}'></div>
              </div>
              <div class='prob-pct' style='color:{label_color}'>{pct:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class='empty-state'>
          <div class='empty-icon'>✍️</div>
          <div class='empty-text' style='color:#475569;font-size:14px;font-weight:500'>Draw a digit on the left<br>to see the prediction</div>
          <div style='margin-top:12px;font-size:11px;color:#334155'>Supports digits 0 through 9</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────
st.markdown("""
<div class='sec-head'>
  <div class='sec-icon'>⚙️</div>
  <div>
    <p class='sec-title'>How It Works</p>
    <p class='sec-sub'>CNN inference pipeline — from canvas to prediction</p>
  </div>
</div>
<div class='info-grid'>
  <div class='info-card'>
    <div class='info-card-icon'>🖊️</div>
    <div class='info-card-title'>Draw</div>
    <div class='info-card-text'>Your stroke is captured as a 300×300 RGBA image on the canvas.</div>
  </div>
  <div class='info-card'>
    <div class='info-card-icon'>🔲</div>
    <div class='info-card-title'>Resize</div>
    <div class='info-card-text'>Image is converted to grayscale and resized to 28×28 — MNIST format.</div>
  </div>
  <div class='info-card'>
    <div class='info-card-icon'>🧠</div>
    <div class='info-card-title'>Predict</div>
    <div class='info-card-text'>CNN model processes the pixel array and outputs probabilities for each digit.</div>
  </div>
  <div class='info-card'>
    <div class='info-card-icon'>📊</div>
    <div class='info-card-title'>Display</div>
    <div class='info-card-text'>Top prediction and all 10 class probabilities shown with confidence level.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class='footer'>
  <div class='footer-brand'>✍️ DigiSense AI — Handwritten Digit Recognition</div>
  <div class='footer-sub'>
    Developed by <b style='color:#a78bfa'>Syed Samiq Abbas Bukhari</b> &nbsp;·&nbsp;
    <b style='color:#38bdf8'>CodeAlpha</b> ML Internship &nbsp;·&nbsp; Task 3 &nbsp;·&nbsp;
    CNN · MNIST · TensorFlow · Streamlit
  </div>
</div>
""", unsafe_allow_html=True)