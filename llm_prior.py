#!/usr/bin/env python3
"""Text-only friction estimates via OpenAI, with validated, settings-aware caching.

Existing prompt-building helpers remain compatible. The live query path refuses
photo references (it does not upload images). Use --query explicitly for an API
request on a cache miss; otherwise the CLI only previews the prompt.
"""
import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

PROMPT_PREFIXES = ("description_", "photo_")
FORBIDDEN_SUBSTRINGS = ("mu_", "interval", "sources", "notes", "s_tab", "se", "n_runs", "in_sim_range", "speed_effect", "load_effect")

PROMPT_TEMPLATE = (
    "You estimate the coefficient of kinetic (sliding) friction between an object and a table surface.\n"
    "Object: {description_object}\n"
    "Table surface: {description_surface}\n"
    "{photo_lines}"
    "Give your median estimate of the kinetic friction coefficient and a 90% interval, as JSON:\n"
    '{{"median": <number>, "low": <number>, "high": <number>, "reasoning": "<one paragraph>"}}\n')

def prompt_fields(row):
    """The subset of a pair row that may reach a prompt: only description_* and photo_* columns."""
    return {k: v for k, v in row.items() if k.startswith(PROMPT_PREFIXES)}

def build_prompt(row):
    """Prompt text from a pair row (any mapping).  Reads only the allowed columns; raises if a value from
    any other column would be interpolated."""
    allowed = prompt_fields(row)
    photos = [v for k, v in sorted(allowed.items()) if k.startswith("photo_") and v]
    text = PROMPT_TEMPLATE.format(description_object=allowed.get("description_object", "").strip() or "(no description)",
                                  description_surface=allowed.get("description_surface", "").strip() or "(no description)",
                                  photo_lines="".join(f"Photo: {p}\n" for p in photos))
    for k, v in row.items():
        if not k.startswith(PROMPT_PREFIXES) and v not in (None, "") and str(v) in text:
            raise ValueError(f"column {k!r} reached the prompt")
    return text

MODEL = "gpt-5.4-mini-2026-03-17"
SCHEMA = {
    "type": "object",
    "properties": {
        "median": {"type": "number"},
        "low": {"type": "number"},
        "high": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["median", "low", "high", "reasoning"],
    "additionalProperties": False,
}


def request_settings():
    """All output-affecting request settings are included in the cache identity."""
    return {
        "model": MODEL,
        "reasoning": {"effort": "none"},
        "max_output_tokens": 600,
        "store": False,
        "tools": [],
        "text": {"format": {"type": "json_schema", "name": "friction_estimate",
                            "strict": True, "schema": SCHEMA}},
    }


def prompt_key(prompt):
    payload = {"cache_version": 2, "endpoint": "https://api.openai.com/v1/responses",
               "prompt": prompt, "settings": request_settings()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def parse_response(raw):
    """Check numeric semantics too: JSON schema alone cannot order the interval."""
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != set(SCHEMA["required"]):
        raise ValueError("Expected median, low, high, and reasoning only")
    for name in ("median", "low", "high"):
        value = data[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not data["low"] <= data["median"] <= data["high"] or data["low"] >= data["high"]:
        raise ValueError("Expected 0 < low <= median <= high, with low < high")
    if not isinstance(data["reasoning"], str) or not data["reasoning"].strip():
        raise ValueError("Expected a nonempty explanation")
    # Do not silently clip estimates to the inference grid or claim calibration.
    return data


def cached_response(prompt, cache_path="llm_cache.json", query=None):
    """Return validated raw JSON; no call on a miss unless query is supplied.

    Old prompt-only entries are retained but not reused. Intended for one writer
    at a time. Cache hits need neither an API key nor the OpenAI package.
    """
    path = Path(cache_path)
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = prompt_key(prompt)
    if key in cache:
        raw = cache[key]["response"]
        parse_response(raw)
        return raw
    if query is None:
        return None
    raw = query(prompt)
    parse_response(raw)  # Never cache malformed estimates.
    cache[key] = {"prompt": prompt, "settings": request_settings(),
                  "created_utc": datetime.now(timezone.utc).isoformat(),
                  "response": raw, "cache_version": 2}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
            temp_name = f.name
            json.dump(cache, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temp_name, path)
    finally:
        if temp_name is not None and os.path.exists(temp_name):
            os.unlink(temp_name)
    return raw


def query_llm(prompt):
    """One API request, no automatic retries; credentials come from the environment."""
    if any(line.startswith("Photo:") for line in prompt.splitlines()):
        raise ValueError("This experiment is text-only; remove photo fields before querying")
    from openai import OpenAI
    client = OpenAI(base_url="https://api.openai.com/v1", max_retries=0, timeout=60.0)
    response = client.responses.create(input=prompt, **request_settings())
    if response.status != "completed" or not response.output_text:
        raise ValueError(f"No complete estimate: status={response.status}; possible refusal or output limit")
    parse_response(response.output_text)
    return response.output_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, help="Object material description")
    parser.add_argument("--surface", required=True, help="Contact surface description")
    parser.add_argument("--query", action="store_true", help="Allow one paid request on a cache miss")
    parser.add_argument("--cache", default="llm_cache_pilot.json", help="Pilot cache, separate from final data")
    args = parser.parse_args()
    if not args.object.strip() or not args.surface.strip():
        parser.error("Both descriptions must be nonempty")
    prompt = build_prompt({"description_object": args.object, "description_surface": args.surface})
    print(f"Model: {MODEL}\nPrompt:\n{prompt}")
    raw = cached_response(prompt, args.cache)
    if raw is not None:
        print("CACHE HIT -- no API request")
    elif args.query:
        print("CACHE MISS -- making one API request", flush=True)
        raw = cached_response(prompt, args.cache, query=query_llm)
    else:
        print("Preview only -- add --query to request and cache an estimate")
        return
    print(json.dumps(parse_response(raw), indent=2))
    print("Pilot estimate only; the reported interval has not been calibrated.")


if __name__ == "__main__":
    main()
