#!/usr/bin/env python3
"""
probes.py -- the prompts of the proposal's calibration probes (E5) and of the direct-speed condition (C1), built from the
same pair descriptions as the friction prompt.  DRAFT WORDINGS: read them, change what should change, then freeze them by
committing this file before any is sent.  Nothing here calls an API; query_claude.py --probe KIND sends a set.

  wording_b       the friction question in other words, same pairs, same answer shape.  Do the answers move with the wording?
  hidden          the object's material withheld ("a block whose contacting face is of an unstated material") on the same
                  surface.  The interval should be wider than for the named pair.  No truth is needed.
  counterfactual  one change to a pair with a known direction (an oil film lowers friction, a rubber pad raises it).  The
                  median should move that way against the unchanged pair.  No truth is needed.
  recite          asks for the handbook's tabulated value for the pair.  A model that recites the table's number has seen it;
                  its friction estimates on those pairs are then recall, not judgement (the memorisation probe).
  speed           C1: the push is described and the model is asked for the paddle's peak speed itself, for one target.
                  No friction model stands between the description and the action.

    python3 evaluation/probes.py                 # how many prompts of each kind, and one example of each
    python3 evaluation/probes.py --kind hidden   # every prompt of one kind
"""
import argparse, json, math
import run_pair_set as R
from llm_prior import SCHEMA as FRICTION_SCHEMA, parse_response as parse_friction
from run_pair_set import SURFACES

X0 = 0.55            # the object's start, m (run_arm_evaluation.X0); a speed prompt states the distance from it to the target

WORDING_B = (
    "An object rests on a flat table and is slid across it.\n"
    "The object: {description_object}\n"
    "The table: {description_surface}\n"
    "What is the coefficient of kinetic friction between them while the object slides? Answer with your median estimate "
    "and the 5th and 95th percentiles of your uncertainty, as JSON:\n"
    '{{"median": <number>, "low": <number>, "high": <number>, "reasoning": "<one paragraph>"}}\n')

FRICTION_PROMPT = (                   # llm_prior.PROMPT_TEMPLATE without the photo line; hidden and counterfactual use it unchanged
    "You estimate the coefficient of kinetic (sliding) friction between an object and a table surface.\n"
    "Object: {description_object}\n"
    "Table surface: {description_surface}\n"
    "Give your median estimate of the kinetic friction coefficient and a 90% interval, as JSON:\n"
    '{{"median": <number>, "low": <number>, "high": <number>, "reasoning": "<one paragraph>"}}\n')

HIDDEN_OBJECT = "A clean, dry block; the material of its contacting face is not stated"

# (suffix added to the object's description, the direction the median should move)
COUNTERFACTUALS = {
    "oil": ("; a thin film of light machine oil has been wiped onto the contacting face", "lower"),
    "ptfe": ("; the contacting face is covered with smooth PTFE (Teflon) tape", "lower"),
    "rubber": ("; a soft rubber pad is glued over the whole contacting face", "higher"),
    "sandpaper": ("; coarse sandpaper is glued over the whole contacting face, grit side out", "higher"),
}

RECITE_PROMPT = (
    "Engineering handbooks tabulate coefficients of kinetic (sliding) friction for pairs of materials.\n"
    "Pair: {object_material} sliding on {surface_material}, clean and dry.\n"
    "If you recall a specific tabulated value for this pair, give it and name the handbook or table. If you do not recall "
    "a specific tabulated value, say so; do not estimate. Answer as JSON:\n"
    '{{"recalls_value": <true or false>, "value": <number or null>, "source": "<handbook or table, or empty>"}}\n')
RECITE_SCHEMA = {"type": "object", "properties": {"recalls_value": {"type": "boolean"}, "value": {"type": ["number", "null"]}, "source": {"type": "string"}},
                 "required": ["recalls_value", "value", "source"], "additionalProperties": False}

SPEED_PROMPT = (
    "A robot pushes an object across a level table with a flat paddle and lets it go; the object then slides to rest.\n"
    "Object: {description_object}\n"
    "Table surface: {description_surface}\n"
    "The paddle starts at rest touching the object, accelerates at 2 m/s^2 to a peak speed, then brakes, with its "
    "deceleration growing at 100 m/s^3. The object leaves the paddle once the paddle slows faster than friction slows the "
    "object, and slides on.\n"
    "The object's centre should come to rest {distance_m:.3f} m beyond where it started.\n"
    "What peak paddle speed should the robot command? Answer as JSON:\n"
    '{{"peak_speed_m_per_s": <number>, "reasoning": "<one paragraph>"}}\n')
SPEED_SCHEMA = {"type": "object", "properties": {"peak_speed_m_per_s": {"type": "number"}, "reasoning": {"type": "string"}},
                "required": ["peak_speed_m_per_s", "reasoning"], "additionalProperties": False}

def parse_recite(raw):
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != set(RECITE_SCHEMA["required"]):
        raise ValueError("Expected recalls_value, value and source only")
    if not isinstance(data["recalls_value"], bool) or not isinstance(data["source"], str):
        raise ValueError("recalls_value must be true or false and source a string")
    v = data["value"]
    if data["recalls_value"]:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ValueError("a recalled value must be a positive number")
    elif v is not None:
        raise ValueError("value must be null when no tabulated value is recalled")
    return data

def parse_speed(raw):
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != set(SPEED_SCHEMA["required"]):
        raise ValueError("Expected peak_speed_m_per_s and reasoning only")
    v = data["peak_speed_m_per_s"]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        raise ValueError("peak_speed_m_per_s must be a positive number")
    if not isinstance(data["reasoning"], str) or not data["reasoning"].strip():
        raise ValueError("Expected a nonempty explanation")
    return data

KINDS = {"wording_b": (FRICTION_SCHEMA, parse_friction), "hidden": (FRICTION_SCHEMA, parse_friction),
         "counterfactual": (FRICTION_SCHEMA, parse_friction), "recite": (RECITE_SCHEMA, parse_recite), "speed": (SPEED_SCHEMA, parse_speed)}

def descriptions(truth_files=R.TRUTH, only=None):
    """{pair: dict(object, surface, description_object, description_surface)}: only what may reach a prompt (never the friction)."""
    import csv
    out = {}
    for path in truth_files:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                pid = r["pair_id"]
                if only and pid not in only:
                    continue
                d_obj, d_surf = r.get("description_object", ""), r.get("description_surface", "")
                if not (d_obj and d_surf):
                    d_obj, d_surf = R.OBJECT, SURFACES[pid]
                out[pid] = dict(object=r["object_material"], surface=r["surface_material"], description_object=d_obj, description_surface=d_surf)
    return out

def build(kind, truth_files=R.TRUTH, only=None, targets=(0.60, 0.78, 0.96)):
    """{prompt id: dict(prompt, pair, ...)} for one kind.  The id names the pair and the variant; the shape matches what
    query_claude.fill takes."""
    if kind not in KINDS:
        raise ValueError(f"kind is one of {sorted(KINDS)}")
    out = {}
    for pid, d in descriptions(truth_files, only).items():
        fields = dict(description_object=d["description_object"], description_surface=d["description_surface"])
        if kind == "wording_b":
            out[f"{pid}/wording_b"] = dict(prompt=WORDING_B.format(**fields), pair=pid)
        elif kind == "hidden":
            out[f"{pid}/hidden"] = dict(prompt=FRICTION_PROMPT.format(**dict(fields, description_object=HIDDEN_OBJECT)), pair=pid, expect="wider")
        elif kind == "counterfactual":
            for name, (suffix, direction) in COUNTERFACTUALS.items():
                out[f"{pid}/{name}"] = dict(prompt=FRICTION_PROMPT.format(**dict(fields, description_object=d["description_object"] + suffix)),
                                           pair=pid, change=name, expect=direction)
        elif kind == "recite":
            out[f"{pid}/recite"] = dict(prompt=RECITE_PROMPT.format(object_material=d["object"], surface_material=d["surface"]), pair=pid)
        else:
            for t in targets:
                out[f"{pid}/speed@{t:.2f}"] = dict(prompt=SPEED_PROMPT.format(distance_m=t - X0, **fields), pair=pid, target=t)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=sorted(KINDS), default=None)
    ap.add_argument("--pairs", nargs="+", default=None)
    a = ap.parse_args()
    for kind in ([a.kind] if a.kind else sorted(KINDS)):
        prompts = build(kind, only=a.pairs)
        print(f"== {kind}: {len(prompts)} prompts")
        for pid, d in (prompts.items() if a.kind else list(prompts.items())[:1]):
            print(f"-- {pid}" + (f"  (expect: {d['expect']})" if "expect" in d else ""))
            print(d["prompt"])
