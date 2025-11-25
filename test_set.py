import os
from PIL import Image, ImageDraw

# Directory for sample test images
output_dir = "sample_test_images"
os.makedirs(output_dir, exist_ok=True)

# Synthetic sample images with labels written on them
samples = {
    "Potato___Early_blight_01.jpg": "Potato Early Blight",
    "Potato___Late_blight_01.jpg": "Potato Late Blight",
    "Tomato___Leaf_Mold_01.jpg": "Tomato Leaf Mold",
    "Tomato___Healthy_01.jpg": "Tomato Healthy",
    "Apple___Black_rot_01.jpg": "Apple Black Rot"
}

# Create synthetic images
for filename, label in samples.items():
    img = Image.new("RGB", (300, 300), color=(120, 180, 90))   # Greenish leaf-like background
    draw = ImageDraw.Draw(img)
    draw.text((20, 140), label, fill=(0, 0, 0))                # Write label text
    img.save(os.path.join(output_dir, filename))

print("Synthetic sample images created in:", output_dir)
