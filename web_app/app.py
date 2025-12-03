import os
import gdown
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import tensorflow_addons as tfa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.makedirs("../models", exist_ok=True)

# Google Drive file IDs
vgg_id = "1M_LOnfmtpUea19SHwP7-ZXGhqR-TBWC_"
unet_id = "1C-Tkhh5gckzeAVf-CvmC86DmY4fSuJwf"

vgg_path = "../models/vgg16_model.h5"
unet_path = "../models/Custom_unet_model.h5"

# Download VGG16 if not exists
if not os.path.exists(vgg_path):
    gdown.download(f"https://drive.google.com/uc?id={vgg_id}", vgg_path, quiet=False)

# Download U-Net if not exists
if not os.path.exists(unet_path):
    gdown.download(f"https://drive.google.com/uc?id={unet_id}", unet_path, quiet=False)

# --------------------------- UI TITLE ---------------------------
st.markdown(
    "<h1 style='text-align: center; color: #FFD700;'>Automate Blood Cell Cancer Classification</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: #00BFFF;'>Upload an image and get predictions from VGG16 & U-Net models</p>",
    unsafe_allow_html=True
)

# --------------------------- LOAD MODELS ---------------------------
Vgg_model = tf.keras.models.load_model('../models/vgg16_model.h5', compile=False)

# Load U-Net with custom_objects for GroupNormalization
Unet_model = tf.keras.models.load_model(
    '../models/Custom_unet_model.h5',
    compile=False,
    custom_objects={'GroupNormalization': tfa.layers.GroupNormalization}
)

# --------------------------- IMAGE UPLOAD ---------------------------
st.sidebar.title("Upload an Image for Prediction")
uploaded_file = st.sidebar.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility='hidden')

predict_btn = st.sidebar.button("Predict")

# --------------------------- HELPER FUNCTION ---------------------------
def prepare_image(img):
    img = img.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# --------------------------- MAIN LOGIC ---------------------------
if predict_btn:
    if uploaded_file is None:
        st.warning("⚠️ Please upload an image first.")
        st.stop()

    img = Image.open(uploaded_file)
    img_prepared = prepare_image(img)

    # ---------------- VGG PREDICTION ----------------
    vgg_probs = Vgg_model.predict(img_prepared)[0]
    vgg_class_idx = np.argmax(vgg_probs)
    vgg_labels = ["Benign", "Early Pre-B", "Pre-B", "Pro-B"]
    vgg_label = vgg_labels[vgg_class_idx]
    vgg_score = vgg_probs[vgg_class_idx] * 100

    # ---------------- UNET PREDICTION ----------------
    unet_probs = Unet_model.predict(img_prepared)[0]
    unet_class_idx = np.argmax(unet_probs)
    unet_labels = ["Benign", "Early Pre-B", "Pre-B", "Pro-B"]
    unet_label = unet_labels[unet_class_idx]
    unet_score = unet_probs[unet_class_idx] * 100

    # ---------------- DISPLAY OUTPUT ----------------
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(img, width=300)
        st.markdown("<p style='text-align: center; color: #FF69B4;'>Uploaded Image</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='text-align: center; color: #1E90FF;'>VGG16 Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size:20px; color: #32CD32;'><b>{vgg_label}</b> ({vgg_score:.2f}%)</p>", unsafe_allow_html=True)

    with col3:
        st.markdown("<h3 style='text-align: center; color: #1E90FF;'>U-Net Prediction</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-size:20px; color: #32CD32;'><b>{unet_label}</b> ({unet_score:.2f}%)</p>", unsafe_allow_html=True)


