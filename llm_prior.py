#!/usr/bin/env python3
"""
llm_prior.py -- the prompt side of the LLM observation model (spec constraint 2 and test 7).

The language model is a sensor: it sees a text description of the object and the table (and photos, when
there are any) and returns a median and a 90% interval for mu.  It must never see ground-truth friction,
intervals, sources or notes, so the prompt builder reads ONLY the `description_*` and `photo_*` columns of
the pair file and refuses anything else.  Prompts are frozen and responses cached before the real measured
file exists; the cache key is the prompt text, so a changed prompt is a new query.

This module is the interface guard; the model call itself belongs to the LLM owner (fill in `query_llm`).
"""
import hashlib, json, os

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

def prompt_key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]

def cached_response(prompt, cache_path="llm_cache.json", query=None):
    """Response for a frozen prompt: from the cache, or from `query(prompt)` and then cached.  Without a
    query function and a cache miss, returns None."""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    key = prompt_key(prompt)
    if key not in cache:
        if query is None:
            return None
        cache[key] = {"prompt": prompt, "response": query(prompt)}
        json.dump(cache, open(cache_path, "w"), indent=1)
    return cache[key]["response"]

def query_llm(prompt):
    """Placeholder for the model call (LLM owner): return the raw response text."""
    raise NotImplementedError("wire the LLM API here; keep prompts frozen and cache every response")
