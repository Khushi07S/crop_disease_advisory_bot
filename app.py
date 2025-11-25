# app.py

import streamlit as st
from PIL import Image
import io

from model import extract_features
from agent import gemini_analyze

st.title("🌿 Hybrid Crop Disease Advisor (EfficientNet-B0 + Gemini)")

uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        with st.spinner("Extracting EfficientNet-B0 features..."):
            features = extract_features(img)

        st.success("Deep learning features extracted successfully.")

        # Convert image to bytes
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        with st.spinner("Gemini analyzing..."):
            result = gemini_analyze(img_bytes, features)

        st.subheader("Advisory")
        st.write(result)
