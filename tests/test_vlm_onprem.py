"""Unit tests for the on-prem VLM module.

These tests do NOT spin up vLLM or talk to a real model; they exercise
the pure-Python pieces (prompt builder, JSON-schema generator, response
parser) so the contract can be validated cheaply.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import load_schema  # noqa: E402
from src.models.vlm_onprem import (  # noqa: E402
    CANNOT_DETERMINE,
    _safe_parse_json,
    build_json_schema,
    build_prompt,
)


def test_build_prompt_lists_only_applicable_attributes() -> None:
    schema = load_schema()
    prompt = build_prompt("stairs", "Stairs", schema)
    assert "attr_number_of_steps" in prompt
    assert "attr_decking_material" not in prompt
    assert CANNOT_DETERMINE in prompt
    assert "Stairs" in prompt


def test_build_json_schema_emits_enum_for_categorical() -> None:
    schema = load_schema()
    js = build_json_schema("trail_bridge", schema)
    assert js["type"] == "object"
    assert "attr_decking_material" in js["properties"]
    value_schema = js["properties"]["attr_decking_material"]["properties"]["value"]
    assert "enum" in value_schema
    assert "Timber" in value_schema["enum"]
    assert CANNOT_DETERMINE in value_schema["enum"]


def test_safe_parse_json_handles_markdown_fences() -> None:
    raw = """```json
{"attr_decking_material": {"value": "Timber", "confidence": 0.9}}
```"""
    parsed = _safe_parse_json(raw)
    assert parsed["attr_decking_material"]["value"] == "Timber"

    parsed2 = _safe_parse_json(
        'Sure! Here is the JSON:\n{"x": {"value": 1, "confidence": 0.5}}\n— end.'
    )
    assert parsed2["x"]["value"] == 1
