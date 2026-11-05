#!/usr/bin/env python3
"""SimReady friction audit (decision 10 input): read the friction fields of
NVIDIA's SimReady Central library and summarise them.

What the library is
-------------------
SimReady Central (https://huggingface.co/spaces/nvidia/simready-central) is a
search front end over OEM-curated Hugging Face datasets. The NVIDIA-authored
library behind the "2,000+ free OpenUSD assets" figure is the dataset
``nvidia/simready-assets`` (CC-BY-4.0, ungated; 2,677 asset folders on
5 Nov 2026). Each asset folder is::

    <asset>/simready_usd/sm_<asset>_01.usd          main stage (binary usdc)
    <asset>/simready_usd/payloads/base.usda         prim structure, variant sets,
                                                    per-mesh material:binding:physics
    <asset>/simready_usd/payloads/Physics/physics.usda
                                                    the neutral physics layer: every prim
                                                    under /RootNode/PhysicsMaterials gets
                                                    PhysicsMaterialAPI with
                                                    physics:staticFriction,
                                                    physics:dynamicFriction,
                                                    physics:restitution, physics:density
    <asset>/simready_usd/payloads/materials.usda    visual materials; the same numbers are
                                                    mirrored as custom pxr:usd:physics_*
                                                    attributes plus a pxr:usd:physics_type
                                                    token ("rubber", "plastic", "steel", ...)
    <asset>/simready_usd/payloads/{PhysX,MuJoCo,Newton}/enabled.usda
                                                    engine overlays (Newton adds rolling and
                                                    torsional friction; PhysX and MuJoCo add
                                                    no friction of their own)
    <asset>/simready_usd/payloads/geometries.usd    the mesh data (large; not needed here)

So one friction pair (static, dynamic) is authored per *physics material*, and a
physics material is bound per mesh. A mesh's friction is the material's; no
per-pair (object x surface) table exists. Each simulator combines the two
bound materials by its own rule when they touch.

What this script does
---------------------
1. Lists the asset folders of ``nvidia/simready-assets`` through the Hub API.
2. Downloads only ``payloads/Physics/physics.usda`` and ``payloads/materials.usda``
   for every asset (a few tens of kB each) into ``--download-dir`` using
   ``huggingface_hub`` (parallel, resumable; files already present are kept).
3. Reads every physics material with ``usd-core`` (``pxr.Usd`` /
   ``pxr.UsdPhysics.MaterialAPI``) and the ``physics_type`` token with
   ``pxr.Sdf`` and writes one row per physics material prim to
   ``simready_friction_values.csv`` in ``--out-dir``.
4. Summarises the distribution, matches the rig's materials
   (``friction_lookup.csv``: nine object bases, three surfaces) to the library
   by material-name keywords, plots ``simready_friction.png`` and writes
   ``simready_audit_summary.json`` and ``simready_audit_tables.md``.

Usage
-----
    .venv/bin/python simready_audit.py --download-dir /path/to/empty/dir --out-dir /path/to/out
    .venv/bin/python simready_audit.py --download-dir ... --out-dir ... --limit 50   # quick check
    .venv/bin/python simready_audit.py --out-dir ... --analyze-only                  # redo tables and figure from the CSV

The figure path defaults to ``simready_friction.png`` in the current directory
(``--figure``). Pipeline code is untouched: this is a read-only study.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

REPO = "nvidia/simready-assets"
PHYSICS_REL = "simready_usd/payloads/Physics/physics.usda"
MATERIALS_REL = "simready_usd/payloads/materials.usda"
BLUE, ORANGE = "#2a78d6", "#eb6834"

# Material-name conventions: physics_mat_m_<opacity>__<base>__<variant>_<ver>
NAME_RE = re.compile(r"^(?:physics_mat_)?(?:m_)?(?P<opacity>[a-z]+)__(?P<base>[A-Za-z0-9]+)__(?P<variant>.+)$")
VER_RE = re.compile(r"_[a-z]\d{2}$")

# Rig materials from friction_lookup.csv (object_material / surface_material).
# A library material has three class keys: the <base> token of its name, its
# pxr:usd:physics_type and its omni:simready:nonvisual:base. A "specific" match
# is a regex on the variant part of the name, accepted only when one of the
# class keys is in specific_classes (so "crate_plywood" on a metal material does
# not count as plywood). With no specific hit the rig material falls back to a
# "class" match: any material whose class key equals one of classes.
RIG_MATERIALS = {
    # nine object bases
    "PLA (3D printed)": dict(specific=r"\bpla\b", specific_classes={"plastic"}, classes={"plastic"}),
    "ABS sheet (extruded)": dict(specific=r"\babs\b", specific_classes={"plastic"}, classes={"plastic"}),
    "furniture felt": dict(specific=r"felt", specific_classes={"fabric", "nylon", "polyester"}, classes={"fabric"}),
    "neoprene rubber 60A": dict(specific=r"neoprene", specific_classes={"rubber"}, classes={"rubber"}),
    "corrugated cardboard": dict(specific=r"corrugat", specific_classes={"cardboard", "paper"}, classes={"cardboard"}),
    "cast acrylic": dict(specific=r"acrylic|plexi|pmma|perspex", specific_classes={"plexiglass", "plastic", "glass", "clear_glass", "frosted_glass"}, classes={"plexiglass"}),
    "oak, grain along travel": dict(specific=r"oak", specific_classes={"wood"}, classes={"wood"}),
    "glazed ceramic tile": dict(specific=r"gloss|glazed|porcelain|tile", specific_classes={"ceramic", "ceramic_glass"}, classes={"ceramic", "ceramic_glass"}),
    "stainless steel 304": dict(specific=r"stainless", specific_classes={"steel", "metal", "stainless_steel_thin"}, classes={"stainless_steel_thin", "steel"}),
    # three surfaces
    "white melamine shelf board": dict(specific=r"melamine|laminate", specific_classes={"wood", "plastic"}, classes={"wood"}),
    "birch plywood, sanded": dict(specific=r"plywood|birch", specific_classes={"wood"}, classes={"wood"}),
    "painted MDF (satin)": dict(specific=r"mdf|fiberboard|fibreboard|particleboard|chipboard", specific_classes={"wood"}, classes={"wood"}),
}
RIG_OBJECTS = list(RIG_MATERIALS)[:9]
RIG_SURFACES = list(RIG_MATERIALS)[9:]

# Friction-combination rules of the engines SimReady ships variants for, plus Drake.
COMBINE_RULES = {
    "Drake 2ab/(a+b)": lambda a, b: 2 * a * b / (a + b) if (a + b) else 0.0,
    "PhysX average": lambda a, b: (a + b) / 2,
    "MuJoCo max": lambda a, b: max(a, b),
}


# ----------------------------------------------------------------------------
# Hub access
# ----------------------------------------------------------------------------
def list_asset_dirs(repo: str) -> list[str]:
    """Top-level folders of the dataset (one per asset)."""
    from huggingface_hub import HfApi

    api = HfApi()
    dirs = []
    for entry in api.list_repo_tree(repo, repo_type="dataset", recursive=False):
        if type(entry).__name__ == "RepoFolder":
            dirs.append(entry.path)
    return sorted(dirs)


def download_one(asset: str, rel: str, local_dir: Path, retries: int = 4):
    """Fetch <asset>/<rel> into local_dir. Returns (status, path_or_message)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    filename = f"{asset}/{rel}"
    for attempt in range(retries):
        try:
            p = hf_hub_download(REPO, filename, repo_type="dataset", local_dir=str(local_dir))
            return "ok", p
        except EntryNotFoundError:
            return "missing", filename
        except Exception as exc:  # 429 / 5xx / network: back off and retry
            if attempt == retries - 1:
                return "error", f"{filename}: {exc}"
            time.sleep(5 * (3 ** attempt))
    return "error", filename


# ----------------------------------------------------------------------------
# USD reading
# ----------------------------------------------------------------------------
def read_physics_layer(path: str, asset: str) -> list[dict]:
    """Every prim with PhysicsMaterialAPI in the neutral physics layer."""
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(path)
    rows = []

    def authored(attr):
        return attr.Get() if attr and attr.HasAuthoredValue() else None

    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.PrimAllPrimsPredicate):
        if not prim.HasAPI(UsdPhysics.MaterialAPI):
            continue
        api = UsdPhysics.MaterialAPI(prim)
        rows.append(
            dict(
                asset=asset,
                material=prim.GetName(),
                prim_path=str(prim.GetPath()),
                static_friction=authored(api.GetStaticFrictionAttr()),
                dynamic_friction=authored(api.GetDynamicFrictionAttr()),
                restitution=authored(api.GetRestitutionAttr()),
                density=authored(api.GetDensityAttr()),
            )
        )
    return rows


def read_materials_layer(path: str) -> dict[str, dict]:
    """physics_type and the mirrored pxr:usd:physics_* values per visual material."""
    from pxr import Sdf

    layer = Sdf.Layer.FindOrOpen(path)
    out = {}
    if layer is None:
        return out

    def default(spec, name):
        a = spec.attributes.get(name)
        return None if a is None else a.default

    def walk(spec):
        for child in spec.nameChildren:
            if child.typeName == "Material":
                labels = default(child, "semantics:labels:material")
                out[child.name] = dict(
                    physics_type=default(child, "pxr:usd:physics_type"),
                    nonvisual_base=default(child, "omni:simready:nonvisual:base"),
                    semantic_labels=";".join(list(labels)) if labels else None,
                    mirror_static=default(child, "pxr:usd:physics_staticFriction"),
                    mirror_dynamic=default(child, "pxr:usd:physics_dynamicFriction"),
                )
            walk(child)

    walk(layer.pseudoRoot)
    return out


def parse_name(material: str) -> tuple[str, str, str]:
    m = NAME_RE.match(material)
    if not m:
        return "", "", material
    variant = VER_RE.sub("", m.group("variant"))
    return m.group("opacity"), m.group("base"), variant


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def collect(args) -> list[dict]:
    t0 = time.time()
    assets = list_asset_dirs(REPO)
    print(f"[list] {len(assets)} asset folders in {REPO} ({time.time() - t0:.0f} s)")
    if args.limit:
        assets = assets[: args.limit]

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(a, rel) for a in assets for rel in (PHYSICS_REL, MATERIALS_REL)]
    status = Counter()
    paths: dict[tuple[str, str], str] = {}
    errors = []
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_one, a, rel, download_dir): (a, rel) for a, rel in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            a, rel = futs[fut]
            st, msg = fut.result()
            status[st] += 1
            if st == "ok":
                paths[(a, rel)] = msg
            elif st == "error":
                errors.append(msg)
            if i % 500 == 0:
                print(f"[download] {i}/{len(jobs)} files, {dict(status)}, {time.time() - t1:.0f} s", flush=True)
    print(f"[download] done: {dict(status)} in {time.time() - t1:.0f} s")
    for e in errors[:20]:
        print("   error:", e)

    t2 = time.time()
    rows: list[dict] = []
    assets_read, assets_missing_physics, assets_no_material = 0, [], []
    for a in assets:
        p = paths.get((a, PHYSICS_REL))
        if p is None:
            assets_missing_physics.append(a)
            continue
        try:
            prims = read_physics_layer(p, a)
        except Exception as exc:
            errors.append(f"{a}: physics read failed: {exc}")
            continue
        assets_read += 1
        if not prims:
            assets_no_material.append(a)
        mats = {}
        mp = paths.get((a, MATERIALS_REL))
        if mp is not None:
            try:
                mats = read_materials_layer(mp)
            except Exception as exc:
                errors.append(f"{a}: materials read failed: {exc}")
        for r in prims:
            opacity, base, variant = parse_name(r["material"])
            visual = r["material"][len("physics_mat_"):] if r["material"].startswith("physics_mat_") else r["material"]
            info = mats.get(visual, {})
            r.update(
                opacity=opacity,
                base=base,
                variant=variant,
                physics_type=info.get("physics_type"),
                nonvisual_base=info.get("nonvisual_base"),
                semantic_labels=info.get("semantic_labels"),
                mirror_static=info.get("mirror_static"),
                mirror_dynamic=info.get("mirror_dynamic"),
            )
            rows.append(r)
    print(f"[read] {assets_read} assets read, {len(rows)} physics material prims, "
          f"{len(assets_missing_physics)} assets without physics.usda, "
          f"{len(assets_no_material)} with no physics material ({time.time() - t2:.0f} s)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["asset", "material", "prim_path", "opacity", "base", "variant", "static_friction",
              "dynamic_friction", "restitution", "density", "physics_type", "nonvisual_base",
              "semantic_labels", "mirror_static", "mirror_dynamic"]
    with open(out_dir / "simready_friction_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    meta = dict(
        repo=REPO,
        assets_listed=len(assets),
        assets_read=assets_read,
        assets_missing_physics=assets_missing_physics,
        assets_no_physics_material=assets_no_material,
        download_status=dict(status),
        errors=errors,
        seconds_list=round(t1 - t0, 1),
        seconds_download=round(t2 - t1, 1),
        seconds_read=round(time.time() - t2, 1),
    )
    with open(out_dir / "simready_collect_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    return rows


def load_rows(out_dir: Path) -> list[dict]:
    rows = []
    with open(out_dir / "simready_friction_values.csv", newline="") as f:
        for r in csv.DictReader(f):
            for k in ("static_friction", "dynamic_friction", "restitution", "density", "mirror_static", "mirror_dynamic"):
                r[k] = float(r[k]) if r[k] not in ("", None, "None") else None
            rows.append(r)
    return rows


def fmt(v):
    if v is None or v == "":
        return "–"
    try:
        return f"{float(v):.3g}"
    except (TypeError, ValueError):
        return str(v)


def class_keys(r: dict) -> set[str]:
    return {k.lower() for k in (r.get("base"), r.get("physics_type"), r.get("nonvisual_base")) if k}


def match_rig(rows: list[dict]) -> list[dict]:
    table = []
    for rig, spec in RIG_MATERIALS.items():
        rx = re.compile(spec["specific"], re.I)
        specific = [r for r in rows if rx.search(r.get("variant") or "") and class_keys(r) & spec["specific_classes"]]
        if specific:
            chosen, strength = specific, "specific"
        else:
            chosen = [r for r in rows if class_keys(r) & spec["classes"]]
            strength = "class" if chosen else "none"
        names = Counter(r["material"] for r in chosen)
        pairs = Counter((r["static_friction"], r["dynamic_friction"]) for r in chosen)
        mode_pair, mode_n = (pairs.most_common(1)[0] if chosen else ((None, None), 0))
        dyn = [r["dynamic_friction"] for r in chosen if r["dynamic_friction"] is not None]
        table.append(dict(
            rig_material=rig,
            role="object" if rig in RIG_OBJECTS else "surface",
            strength=strength,
            matched_on=spec["specific"] if strength == "specific" else "/".join(sorted(spec["classes"])) if strength == "class" else "",
            n_prims=len(chosen),
            n_assets=len({r["asset"] for r in chosen}),
            n_distinct_materials=len(names),
            example_materials=[n for n, _ in names.most_common(3)],
            values=sorted(((s, d, c) for (s, d), c in pairs.items()), key=lambda x: -x[2]),
            mode_static=mode_pair[0], mode_dynamic=mode_pair[1], mode_share=(mode_n / len(chosen)) if chosen else None,
            dynamic_median=statistics.median(dyn) if dyn else None,
            dynamic_min=min(dyn) if dyn else None, dynamic_max=max(dyn) if dyn else None,
        ))
    return table


def read_lookup(path: Path) -> list[dict]:
    """Rig rows of friction_lookup.csv (by column name)."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("role") == "rig"]


def pair_table(match_table: list[dict], lookup: list[dict]) -> list[dict]:
    """A C2-SimReady value for every rig pair: each side's class value combined by each engine's rule."""
    by_name = {m["rig_material"]: m for m in match_table}
    out = []
    for row in lookup:
        o, s = by_name.get(row["object_material"]), by_name.get(row["surface_material"])
        if not o or not s or o["mode_dynamic"] is None or s["mode_dynamic"] is None:
            out.append(dict(pair_id=row["pair_id"], object=row["object_material"], surface=row["surface_material"],
                            mu_obj=None, mu_surf=None, combos={}, lookup=float(row["mu_k_nominal"]) if row.get("mu_k_nominal") else None))
            continue
        a, b = o["mode_dynamic"], s["mode_dynamic"]
        out.append(dict(pair_id=row["pair_id"], object=row["object_material"], surface=row["surface_material"],
                        mu_obj=a, mu_surf=b, combos={k: round(f(a, b), 3) for k, f in COMBINE_RULES.items()},
                        lookup=float(row["mu_k_nominal"]) if row.get("mu_k_nominal") else None))
    return out


def analyze(rows: list[dict], args) -> dict:
    out_dir = Path(args.out_dir)
    r4 = lambda v: None if v is None else round(v, 4)
    for r in rows:
        r["static_friction"] = r4(r["static_friction"])
        r["dynamic_friction"] = r4(r["dynamic_friction"])
        if not r.get("base") or not r.get("variant"):
            r["opacity"], r["base"], r["variant"] = parse_name(r["material"])
    n_prims = len(rows)
    assets = sorted({r["asset"] for r in rows})
    n_assets = len(assets)
    static_c = Counter(r["static_friction"] for r in rows)
    dynamic_c = Counter(r["dynamic_friction"] for r in rows)
    pair_c = Counter((r["static_friction"], r["dynamic_friction"]) for r in rows)
    equal = sum(1 for r in rows if r["static_friction"] == r["dynamic_friction"])
    mode_static, mode_static_n = static_c.most_common(1)[0]
    mode_dynamic, mode_dynamic_n = dynamic_c.most_common(1)[0]
    by_asset = defaultdict(list)
    for r in rows:
        by_asset[r["asset"]].append(r)
    assets_all_modal = sum(1 for a in by_asset.values() if all(r["dynamic_friction"] == mode_dynamic for r in a))
    assets_one_value = sum(1 for a in by_asset.values() if len({r["dynamic_friction"] for r in a}) == 1)
    distinct_names = Counter(r["material"] for r in rows)
    # cumulative share of the most common (static, dynamic) pairs
    cum, cum_table = 0, []
    for (s, d), c in pair_c.most_common(12):
        cum += c
        cum_table.append((s, d, c, c / n_prims, cum / n_prims))
    # class purity: one value per class?
    type_c = Counter(r["physics_type"] or "(none)" for r in rows)
    type_vals = defaultdict(Counter)
    for r in rows:
        type_vals[r["physics_type"] or "(none)"][(r["static_friction"], r["dynamic_friction"])] += 1
    purity = []
    for t_, c in type_c.most_common():
        (s, d), m = type_vals[t_].most_common(1)[0]
        purity.append((t_, c, s, d, m / c, len(type_vals[t_])))
    classes_pure = sum(1 for t_, c, s, d, sh, k in purity if c >= 10 and sh >= 0.95)
    classes_big = sum(1 for t_, c, s, d, sh, k in purity if c >= 10)
    base_c = Counter(r["base"] or "(unparsed)" for r in rows)
    base_vals = defaultdict(Counter)
    for r in rows:
        base_vals[r["base"] or "(unparsed)"][(r["static_friction"], r["dynamic_friction"])] += 1
    nv_c = Counter(r["nonvisual_base"] or "(none)" for r in rows)
    mirror_mismatch = sum(
        1 for r in rows
        if r.get("mirror_dynamic") is not None and r["dynamic_friction"] is not None
        and abs(r["mirror_dynamic"] - r["dynamic_friction"]) > 1e-6
    )
    match_table = match_rig(rows)
    lookup = read_lookup(Path(args.lookup))
    pairs = pair_table(match_table, lookup)

    summary = dict(
        n_assets_with_physics_materials=n_assets,
        n_physics_material_prims=n_prims,
        n_distinct_material_names=len(distinct_names),
        static_value_counts={str(k): v for k, v in sorted(static_c.items(), key=lambda t: (t[0] is None, t[0]))},
        dynamic_value_counts={str(k): v for k, v in sorted(dynamic_c.items(), key=lambda t: (t[0] is None, t[0]))},
        pair_counts={f"{s},{d}": c for (s, d), c in pair_c.most_common()},
        top_pairs_cumulative=cum_table,
        static_equals_dynamic=equal,
        mode_static=mode_static, mode_static_share=mode_static_n / n_prims,
        mode_dynamic=mode_dynamic, mode_dynamic_share=mode_dynamic_n / n_prims,
        assets_all_prims_at_mode=assets_all_modal,
        assets_single_dynamic_value=assets_one_value,
        mirror_mismatch=mirror_mismatch,
        physics_type_counts=dict(type_c.most_common()),
        physics_type_purity=purity,
        classes_with_10_prims=classes_big, classes_single_value_95pct=classes_pure,
        base_counts=dict(base_c.most_common()),
        nonvisual_base_counts=dict(nv_c.most_common()),
        rig_matches=match_table,
        rig_pairs=pairs,
    )
    with open(out_dir / "simready_audit_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)

    md = []
    md.append(f"Physics material prims: {n_prims} in {n_assets} assets; {len(distinct_names)} distinct material names.\n")
    md.append("| value | dynamic: prims | share | static: prims | share |\n|---|---|---|---|---|")
    keys = sorted(set(static_c) | set(dynamic_c), key=lambda k: (k is None, k))
    for k in keys:
        md.append(f"| {fmt(k)} | {dynamic_c.get(k, 0)} | {dynamic_c.get(k, 0) / n_prims:.1%} | {static_c.get(k, 0)} | {static_c.get(k, 0) / n_prims:.1%} |")
    md.append("")
    md.append("| (static, dynamic) | prims | share | cumulative |\n|---|---|---|---|")
    for s, d, c, sh, cu in cum_table:
        md.append(f"| ({fmt(s)}, {fmt(d)}) | {c} | {sh:.1%} | {cu:.1%} |")
    md.append("")
    md.append(f"static == dynamic on {equal} of {n_prims} prims ({equal / n_prims:.1%}); "
              f"mode dynamic {mode_dynamic} on {mode_dynamic_n} prims ({mode_dynamic_n / n_prims:.1%}); "
              f"{assets_all_modal} of {n_assets} assets carry the mode on every material ({assets_all_modal / n_assets:.1%}); "
              f"{assets_one_value} of {n_assets} assets have a single dynamic value ({assets_one_value / n_assets:.1%}); "
              f"materials.usda mirror disagrees on {mirror_mismatch} prims; "
              f"{classes_pure} of {classes_big} physics_type classes with 10+ prims have one (static, dynamic) pair on 95%+ of their prims.\n")
    md.append("| physics_type | prims | modal (static, dynamic) | share of class | distinct pairs |\n|---|---|---|---|---|")
    for t_, c, s, d, sh, k in purity:
        md.append(f"| {t_} | {c} | ({fmt(s)}, {fmt(d)}) | {sh:.1%} | {k} |")
    md.append("")
    md.append("| rig material | role | match | matched on | materials (assets, prims) | modal (static, dynamic) | share | dynamic min–max | examples |\n|---|---|---|---|---|---|---|---|---|")
    for m in match_table:
        if m["n_prims"]:
            md.append(f"| {m['rig_material']} | {m['role']} | {m['strength']} | `{m['matched_on']}` | {m['n_distinct_materials']} ({m['n_assets']}, {m['n_prims']}) | "
                      f"({fmt(m['mode_static'])}, {fmt(m['mode_dynamic'])}) | {m['mode_share']:.0%} | {fmt(m['dynamic_min'])}–{fmt(m['dynamic_max'])} | {', '.join(m['example_materials'])} |")
        else:
            md.append(f"| {m['rig_material']} | {m['role']} | none |  | 0 |  |  |  |  |")
    md.append("")
    if pairs:
        md.append("| pair | object | surface | μ_obj | μ_surf | " + " | ".join(COMBINE_RULES) + " | lookup μ_k |\n|---|---|---|---|---|" + "---|" * len(COMBINE_RULES) + "---|")
        for q in pairs:
            combos = " | ".join(fmt(q["combos"].get(k)) for k in COMBINE_RULES)
            md.append(f"| {q['pair_id']} | {q['object']} | {q['surface']} | {fmt(q['mu_obj'])} | {fmt(q['mu_surf'])} | {combos} | {fmt(q['lookup'])} |")
    with open(out_dir / "simready_audit_tables.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    return summary


def plot(rows: list[dict], summary: dict, figure: Path):
    """One panel: how many physics materials carry each authored static / dynamic value.
    The values are discrete (27 of them), so each value gets one evenly spaced slot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    static_c = Counter(r["static_friction"] for r in rows if r["static_friction"] is not None)
    dynamic_c = Counter(r["dynamic_friction"] for r in rows if r["dynamic_friction"] is not None)
    values = sorted(set(static_c) | set(dynamic_c))
    n = len(rows)
    pos = list(range(len(values)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    ax.bar([x - w / 2 - 0.02 for x in pos], [static_c.get(v, 0) for v in values], width=w, color=BLUE,
           label="static friction (physics:staticFriction)", linewidth=0)
    ax.bar([x + w / 2 + 0.02 for x in pos], [dynamic_c.get(v, 0) for v in values], width=w, color=ORANGE,
           label="dynamic friction (physics:dynamicFriction)", linewidth=0)
    for x, v in zip(pos, values):  # direct labels only on the values that carry 3% or more of the library;
        # each label is anchored to its own side of the slot so the two never collide
        cs, cd = static_c.get(v, 0), dynamic_c.get(v, 0)
        if cs >= 0.03 * n:
            ax.text(x - 0.03, cs + n * 0.004, f"{cs}", ha="right", va="bottom", fontsize=6.5, color="#444444")
        if cd >= 0.03 * n:
            ax.text(x + 0.03, cd + n * 0.004, f"{cd}", ha="left", va="bottom", fontsize=6.5, color="#444444")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{v:g}" for v in values], fontsize=7, rotation=45, ha="right")
    ax.set_xlabel("friction coefficient authored on the physics material (one slot per distinct value)")
    ax.set_ylabel("physics material prims")
    ax.set_title(f"SimReady friction values: {n} physics materials in {summary['n_assets_with_physics_materials']} assets "
                 f"({REPO}, read 5 Nov 2026)", fontsize=9.5, loc="left")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#bbbbbb")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_xlim(-0.7, len(values) - 0.3)
    fig.tight_layout()
    fig.savefig(figure)
    print(f"[figure] {figure}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--download-dir", help="empty directory for the downloaded .usda layers")
    ap.add_argument("--out-dir", required=True, help="where the CSV, JSON and table files go")
    ap.add_argument("--figure", default="simready_friction.png")
    ap.add_argument("--lookup", default="friction_lookup.csv", help="rig pairs to combine the matched values for")
    ap.add_argument("--limit", type=int, default=0, help="read only the first N assets")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--analyze-only", action="store_true", help="reuse the CSV in --out-dir")
    args = ap.parse_args()
    t0 = time.time()
    if args.analyze_only:
        rows = load_rows(Path(args.out_dir))
    else:
        if not args.download_dir:
            ap.error("--download-dir is required unless --analyze-only")
        rows = collect(args)
    if not rows:
        print("no physics materials read; nothing to summarise")
        sys.exit(1)
    summary = analyze(rows, args)
    plot(rows, summary, Path(args.figure))
    print(f"[done] {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
