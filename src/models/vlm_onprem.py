"""On-prem VLM attribute extraction.

Talks to a vLLM-served model (Qwen2.5-VL, InternVL3, LLaVA-NeXT, etc.)
over an OpenAI-compatible HTTP API.  Images stay on the box — no
external APIs are called.

Pipeline:
1. Group test rows by ``asset_id``.
2. Build one prompt per asset whose body lists the attributes that
   apply to the asset's ``profile_name`` and the candidate vocabularies
   (read from :mod:`src.data.schema`).
3. Attach every available image of the asset to the prompt.
4. Send to the vLLM endpoint with ``response_format={"type": "json_object"}``
   (and ``extra_body={"guided_json": schema}`` when supported) so the
   model returns a JSON object we can parse deterministically.
5. Map each predicted label back to the canonical schema vocabulary
   via the alias tables; the ``"Cannot Determine"`` escape hatch becomes
   NaN.

The same module is reused for Qwen2.5-VL, InternVL3, LLaVA-NeXT, and
any other OpenAI-API-compatible VLM; the only thing that changes is
``--model`` and ``--base-url``.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from src.data.schema import AttributeKind, Schema, load_schema
from src.data.splits import absolute_image_path

logger = logging.getLogger(__name__)

CANNOT_DETERMINE = "Cannot Determine"
MAX_IMAGE_DIM = 1024


class VLMBackend(Protocol):
    """Minimal protocol the pipeline needs from a backend."""

    def predict(self, prompt: str, images: list[Image.Image]) -> dict[str, Any]: ...


@dataclass
class OpenAICompatibleBackend:
    """vLLM-served Qwen2.5-VL / InternVL3 / LLaVA-NeXT etc.

    Talks to the OpenAI-compatible HTTP API.  vLLM honours the
    ``guided_json`` extra body for fully-structured output; we still
    fall back to free-form JSON parsing for backends that do not.
    """

    base_url: str
    model: str
    api_key: str = "EMPTY"
    temperature: float = 0.0
    max_tokens: int = 1024
    guided_json: bool = True

    def __post_init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def predict(
        self,
        prompt: str,
        images: list[Image.Image],
        json_schema: dict | None = None,
    ) -> dict[str, Any]:
        content: list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _to_data_url(img)},
                }
            )

        extra_body: dict = {}
        response_format: dict | None = {"type": "json_object"}
        if self.guided_json and json_schema is not None:
            extra_body["guided_json"] = json_schema
            response_format = None  # vLLM's guided_json replaces response_format

        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if extra_body:
            kwargs["extra_body"] = extra_body

        resp = self._client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or "{}"
        return _safe_parse_json(raw)


def _to_data_url(img: Image.Image) -> str:
    img = ImageOps.exif_transpose(img.convert("RGB"))
    img.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_JSON_BLOCK = re.compile(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", re.DOTALL)


def _safe_parse_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(raw)
        if match is None:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def build_prompt(
    asset_type_id: str,
    profile_name: str,
    schema: Schema,
) -> str:
    attrs = schema.attributes_for_asset_type(asset_type_id)
    lines = [
        "You are an expert reviewer of BC Parks infrastructure photos.",
        "",
        f"The images below are all of a single asset of type \"{profile_name}\".",
        "Use ALL of them to predict each requested attribute.",
        "",
        "Attributes to predict:",
    ]
    for col in attrs:
        attr = schema.attributes[col]
        if attr.kind in {AttributeKind.CATEGORICAL, AttributeKind.BOOLEAN}:
            vocab = " | ".join(attr.values) if attr.values else "any reasonable category"
            lines.append(f"- {col}: one of [ {vocab} ] (or {CANNOT_DETERMINE!r})")
        elif attr.kind == AttributeKind.NUMERIC:
            unit = f" in {attr.units}" if attr.units else ""
            lines.append(f"- {col}: numeric value{unit} (or {CANNOT_DETERMINE!r})")
        elif attr.kind == AttributeKind.COUNT:
            lines.append(f"- {col}: integer count (or {CANNOT_DETERMINE!r})")
        elif attr.kind == AttributeKind.ORDINAL_BIN:
            lines.append(f"- {col}: pick an ordinal bin (or {CANNOT_DETERMINE!r})")
    lines.extend(
        [
            "",
            "Be conservative: if an image does not give you enough information to be confident, ",
            f"return value={CANNOT_DETERMINE!r} and confidence=0.0.",
            "",
            "Respond with a single JSON object of the form:",
            "{",
        ]
        + [f'  "{c}": {{"value": <prediction>, "confidence": <0..1>}},' for c in attrs]
        + [
            "}",
            "Return ONLY the JSON object, no commentary, no markdown.",
        ]
    )
    return "\n".join(lines)


def build_json_schema(asset_type_id: str, schema: Schema) -> dict[str, Any]:
    """Generate a JSON schema vLLM can enforce via ``guided_json``."""
    attrs = schema.attributes_for_asset_type(asset_type_id)
    properties: dict[str, Any] = {}
    for col in attrs:
        attr = schema.attributes[col]
        if attr.kind in {AttributeKind.CATEGORICAL, AttributeKind.BOOLEAN}:
            value_schema = {"type": "string", "enum": [*attr.values, CANNOT_DETERMINE]}
        elif attr.kind == AttributeKind.NUMERIC:
            value_schema = {"type": ["number", "string"]}
        elif attr.kind == AttributeKind.COUNT:
            value_schema = {"type": ["integer", "string"]}
        else:
            value_schema = {"type": "string"}
        properties[col] = {
            "type": "object",
            "properties": {
                "value": value_schema,
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["value", "confidence"],
        }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _normalise_value(
    schema: Schema, attribute_column: str, raw_value: Any
) -> Any:
    if raw_value is None:
        return np.nan
    if isinstance(raw_value, str) and raw_value.strip().lower() == CANNOT_DETERMINE.lower():
        return np.nan
    attr = schema.attributes[attribute_column]
    if attr.kind in {AttributeKind.CATEGORICAL, AttributeKind.BOOLEAN}:
        return attr.normalise_label(raw_value)
    if attr.kind in {AttributeKind.NUMERIC, AttributeKind.COUNT}:
        try:
            return float(raw_value)
        except (TypeError, ValueError):
            return np.nan
    return raw_value


def _load_asset_images(
    image_paths: Iterable[str], repo_root: Path | None = None, max_images: int = 4
) -> list[Image.Image]:
    out: list[Image.Image] = []
    for p in image_paths:
        if len(out) >= max_images:
            break
        full = absolute_image_path(str(p), repo_root=repo_root)
        if not full.exists():
            continue
        try:
            img = Image.open(full).convert("RGB")
            out.append(ImageOps.exif_transpose(img))
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load %s: %s", full, exc)
    return out


def _resolve_asset_type(profile_name: str, schema: Schema) -> str | None:
    return schema.asset_type_for_profile_name(profile_name)


def predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    schema: Schema,
    *,
    backend: VLMBackend,
    max_images_per_asset: int = 4,
    use_guided_json: bool = True,
    repo_root: Path | None = None,
    max_assets: int | None = None,
) -> pd.DataFrame:
    """Predict per asset, then expand back to one row per image.

    ``train_df`` is unused (zero-shot VLM).
    """
    del train_df

    rows: list[dict] = []
    asset_iter = list(test_df.groupby("asset_id", sort=False))
    if max_assets is not None:
        asset_iter = asset_iter[:max_assets]

    for asset_id, group in asset_iter:
        profile_name = group["profile_name"].iloc[0]
        asset_type_id = _resolve_asset_type(profile_name, schema)
        if asset_type_id is None:
            continue

        images = _load_asset_images(
            group["image_path"].tolist(),
            repo_root=repo_root,
            max_images=max_images_per_asset,
        )
        if not images:
            continue

        prompt = build_prompt(asset_type_id, profile_name, schema)
        json_schema = (
            build_json_schema(asset_type_id, schema) if use_guided_json else None
        )
        try:
            if json_schema is not None and hasattr(backend, "predict_with_schema"):
                response = backend.predict_with_schema(  # type: ignore[attr-defined]
                    prompt, images, json_schema
                )
            elif json_schema is not None and isinstance(backend, OpenAICompatibleBackend):
                response = backend.predict(prompt, images, json_schema=json_schema)
            else:
                response = backend.predict(prompt, images)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM call failed for asset %s: %s", asset_id, exc)
            continue

        applicable_attrs = schema.attributes_for_asset_type(asset_type_id)
        per_attr_value: dict[str, Any] = {}
        per_attr_conf: dict[str, float] = {}
        for col in applicable_attrs:
            entry = response.get(col, {}) or {}
            value = entry.get("value") if isinstance(entry, dict) else entry
            confidence = entry.get("confidence") if isinstance(entry, dict) else 0.0
            per_attr_value[col] = _normalise_value(schema, col, value)
            try:
                per_attr_conf[col] = float(confidence or 0.0)
            except (TypeError, ValueError):
                per_attr_conf[col] = 0.0

        for _, row in group.iterrows():
            out_row = {"image_path": row["image_path"]}
            for col in schema.attribute_columns():
                if col in applicable_attrs:
                    out_row[col] = per_attr_value.get(col, np.nan)
                else:
                    out_row[col] = np.nan
                out_row[f"{col}__confidence"] = per_attr_conf.get(col, 0.0)
            rows.append(out_row)

    return pd.DataFrame(rows)
