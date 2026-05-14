from .config import openai_client, gemini_client, MODEL_FAMILIES
from google.genai import types
import base64

# -------------------------------------------
# Gemini formatting
# -------------------------------------------
def build_gemini_contents(prompt, images):
    parts = [prompt]
    for img in images:
        parts.append(
            types.Part.from_bytes(
                data=base64.b64decode(img["b64"]),
                mime_type=img["mime"]
            )
        )
    return parts

# -------------------------------------------
# OpenAI / GitHub formatting
# -------------------------------------------
def build_openai_messages(prompt, images):
    content = []
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img['mime']};base64,{img['b64']}"
            }
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]

# -------------------------------------------
# Model router
# -------------------------------------------
def detect_model_family(model_name):
    for family, models in MODEL_FAMILIES.items():
        if model_name in models:
            return family
    return "openai"  # default fallback

# -------------------------------------------
# General predict function
# -------------------------------------------
def run_model(model_name, prompt, images):
    family = detect_model_family(model_name)

    if family == "gemini":
        contents = build_gemini_contents(prompt, images)
        response = gemini_client.models.generate_content(
            model=model_name,
            contents=contents
        )
        return response.text

    else:
        messages = build_openai_messages(prompt, images)
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content