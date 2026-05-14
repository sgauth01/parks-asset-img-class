import os
import base64
import json
import re
from dotenv import load_dotenv
from google import genai
from openai import OpenAI

load_dotenv()

# -----------------------------
# OpenAI / GitHub Client
# -----------------------------
openai_client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN")
)

# -----------------------------
# Gemini Client
# -----------------------------
gemini_client = genai.Client()

# -----------------------------
# Supported Models
# -----------------------------
MODEL_FAMILIES = {
    "phi": ["Phi-4-multimodal-instruct"],
    "llama": ["Llama-3.2-11B-Vision-Instruct"],
    "gpt": ["gpt-4o"],
    "gemini": ["gemini-3-flash-preview", "gemma-4-26b-a4b-it"],
}