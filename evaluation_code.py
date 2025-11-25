import os
import time
import re
from PIL import Image

from model import extract_features
from agent import gemini_analyze
import google.genai.errors as genai_errors

# Retry-safe Gemini call
def safe_gemini_call(image_bytes, features, retries=5):
    for attempt in range(retries):
        try:
            return gemini_analyze(image_bytes, features)
        except genai_errors.ServerError as e:
            if "503" in str(e):
                wait = 2 ** attempt
                print(f"⚠️ Gemini overloaded (503). Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                raise e
    raise RuntimeError("Gemini failed after multiple retries.")

# Folder with sample images
test_folder = "sample_test_images"

patterns = [
    "early blight",
    "late blight",
    "leaf mold",
    "black rot",
    "healthy"
]

def extract_predicted_label(text):
    text = text.lower()
    for p in patterns:
        if p in text:
            return p
    return "unknown"

results = []
correct = 0
total = 0

for filename in os.listdir(test_folder):
    if not filename.endswith(".jpg"):
        continue
    
    true_label = filename.split("___")[1].split(".")[0].replace("_", " ").lower()
    true_label = re.sub(r"\d+", "", true_label).strip()
    img_path = os.path.join(test_folder, filename)
    img = Image.open(img_path)

    # Reduce size for faster Gemini calls
    img = img.resize((512, 512))

    features = extract_features(img)

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    # Safe Gemini call
    response = safe_gemini_call(img_bytes, features)

    pred_label = extract_predicted_label(response)

    results.append([filename, true_label, pred_label])

    if true_label in pred_label:
        correct += 1
    total += 1

    time.sleep(1.5)  # prevent overload

# Final accuracy
print("\nRESULTS TABLE:")
for r in results:
    print(r)

print("\nFinal Accuracy:", correct / total)
