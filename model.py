# model.py

import numpy as np
from PIL import Image
import tensorflow as tf

# Load EfficientNet-B0 (ImageNet pre-trained)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    pooling="avg",
    input_shape=(224, 224, 3)
)

def preprocess_image(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, 0)

def extract_features(img: Image.Image):
    """Returns 1280-dim EfficientNet feature vector."""
    x = preprocess_image(img)
    features = base_model.predict(x)[0]
    return features
