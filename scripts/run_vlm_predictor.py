#!/usr/bin/env python3
"""
Batch VLM predictor script.

Usage:
    python scripts/run_vlm_predictor.py \
        --input data/processed/train/attr_number_of_steps_train.csv \
        --output results/vlm_predictions_stairs_gemma.csv \
        --model gemini-3-flash-preview \
        --limit 5
"""

import argparse
import os
import sys
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# ---------------------------------------------------------------------
# Ensure project root is in import path
# ---------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.vlm.predictors import predict_asset_attributes

# ---------------------------------------------------------------------
# VLM prompt
# ---------------------------------------------------------------------
PROMPT_TEMPLATE = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single stair asset, identify the most likely
    attribute values. For each of the following attributes, the possible values are
    given below. Predict exactly ONE value from the listed options for each
    attribute, and provide a confidence score (0.0-1.0) for each prediction.

    Attributes to predict:
    - fall_height: low (<0.5m) | medium (0.5m-1.2m) | high (>1.2m)
    - has_pedestrian_railing: 2 railings | 1 railing | no railings
    - material: PVC | Gravel | Natural Surface | Earth-filled | Aluminum | 
                Metal | Steel | Rock/Stone | Concrete | Box Step | Timber/Wood
    - number_of_steps: <integer>
    - structure_position: Elevated | At-Grade | Other

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "<attribute_key>": {
        "value": "<predicted value or 'unable to determine'>",
        "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine an attribute from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

# ---------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------
def run_batch(input_path, output_path, model_name, limit=None):
    print(f"Loading input from: {input_path}")
    df = pd.read_csv(input_path)

    if "asset_id" not in df.columns:
        raise ValueError("Input file must contain an 'asset_id' column")
    
    unique_asset_ids = df['asset_id'].unique()

    if limit:
        unique_asset_ids = unique_asset_ids[:limit]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Running model: {model_name}")
    print(f"Total assets to process: {len(unique_asset_ids)}")
    print(f"Writing results to: {output_path}")

    results = []

    for asset_id in tqdm(unique_asset_ids):
        asset_df = df[df["asset_id"] == asset_id]
        
        result = None

        try:
            result = predict_asset_attributes(
                asset_id=int(asset_id),
                df=asset_df,
                model_name=model_name,
                prompt=PROMPT_TEMPLATE
            )

            out = {
                "asset_id": int(asset_id),
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "response": result.get("response"),
            }

        except Exception as e:
            out = {
                "asset_id": int(asset_id),
                "timestamp": datetime.now().isoformat(),
                "model": model_name,
                "error": str(e),
            }
        
        parsed = None
        if result is not None:
            parsed = result.get("response")

            if isinstance(parsed, dict) and "predictions" in parsed:
                preds = parsed["predictions"]

                for attr, val in preds.items():
                    out[f"{attr}_value"] = val.get("value")
                    out[f"{attr}_confidence"] = val.get("confidence")

        else:
            out["parse_error"] = True
            
        results.append(out)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_path, index=False)

    print("✅ Done! Processed", len(unique_asset_ids), "assets.")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch VLM predictor script.")
    parser.add_argument("--input", required=True, help="Path to training data (CSV)")
    parser.add_argument("--output", required=True, help="Path to store JSONL results")
    parser.add_argument("--model", required=True, help="VLM model name")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for debugging")

    args = parser.parse_args()

    run_batch(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        limit=args.limit
    )