# agent.py

import os
from dotenv import load_dotenv

from google import genai
from google.genai import types

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("❌ GEMINI_API_KEY not found in .env file!")

# Initialize Gemini
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash"


def gemini_analyze(image_bytes: bytes, features):
    """
    Hybrid Agent:
    - EfficientNet-B0 vector (first 40 dims)
    - Gemini Vision (image)
    """

    # Convert EfficientNet features into readable string (first 40)
    feature_str = ", ".join([f"{x:.4f}" for x in features[:40]])

    # Instruction for Gemini
    system_prompt = f"""
You are an agricultural crop disease expert.

You receive:
1. A plant leaf image
2. EfficientNet-B0 feature vector (first 40 dims)

Your tasks:
- Analyze the leaf image for visible disease symptoms.
- Use feature vector as supporting evidence.
- Predict the most likely disease(s).
- Provide:
    • Reasoning based on symptoms
    • Immediate safe management
    • Long-term IPM steps
    • Uncertainty notes if unsure

Do NOT recommend pesticide brand names.
"""

    # Correct image part
    image_part = types.Part(
        inline_data=types.Blob(
            mime_type="image/jpeg",
            data=image_bytes
        )
    )

    # Correct text part
    features_part = types.Part(
        text=f"EfficientNet feature vector (first 40 dims): {feature_str}"
    )

    # Build content message
    contents = [
        types.Content(
            role="user",
            parts=[image_part, features_part]
        )
    ]

    # Call Gemini
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.3
        )
    )

    return response.text or "No response generated."
