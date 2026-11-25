#!/usr/bin/env python3
"""
make_handbook_truth.py -- the handbook pairs of the lookup table as a truth file for the simulator.

Y01-Y03 are all stainless steel on a sheet, friction 0.14-0.26.  To test the methods on other materials before the rig
data exists, this writes data/ground_truth_handbook.csv: the 20 handbook pairs of friction_lookup.csv (L01-L20: metals
on steel and cast iron, engineering plastics on steel, oak, glass, graphite; friction 0.14-0.49), with the lookup's
nominal value as the simulator's truth and a prompt description for each pair.  Same 21 columns as
ground_truth_published.csv (docs/measured_friction_interface.md).

Two things follow from where the truth comes from, and every result on these pairs has to carry them:
  * the lookup baseline holds the answer (its nominal IS the truth here), so on these pairs it is a ceiling, not a
    competitor; the comparison is generic prior against LLM prior;
  * the LLM may recall the same handbooks, so its error here is a lower bound on its error for pairs no table lists.

The descriptions name the materials and the condition the handbook values are for (clean and dry) and nothing else: no
coefficient, no source.  They are prompt text: edit DESCRIPTIONS and rerun to change them (cached LLM answers are keyed
by the prompt, so a changed description is a cache miss, never a silent reuse).

    python3 friction/make_handbook_truth.py            # -> data/ground_truth_handbook.csv
"""
import argparse, csv
from paths import DATA

FIELDS = ["pair_id", "object_id", "surface_id", "object_material", "surface_material", "mu_k_nominal", "s_tab_log", "interval90_low",
          "interval90_high", "sources", "n_runs", "se", "speed_effect", "load_effect", "temp_C", "RH_pct", "in_sim_range",
          "description_object", "description_surface", "photo_object", "photo_surface"]

class _Block:
    @staticmethod
    def format(material):
        return f"A clean, dry block with {'an' if material[0].lower() in 'aeiou' and not material.startswith('UHMW') else 'a'} {material} contacting face"
_BLOCK = _Block
_PLATE = "A clean, dry flat plate of {}"
DESCRIPTIONS = {     # pair -> (object, table surface)
    "L01": (_BLOCK.format("hardened-steel"), _PLATE.format("mild steel")),
    "L02": (_BLOCK.format("aluminium"), _PLATE.format("mild steel")),
    "L03": (_BLOCK.format("copper"), _PLATE.format("mild steel")),
    "L04": (_BLOCK.format("brass"), _PLATE.format("mild steel")),
    "L05": (_BLOCK.format("phenolic-laminate (laminated plastic)"), _PLATE.format("mild steel")),
    "L06": (_BLOCK.format("acetal (POM, Delrin) plastic"), _PLATE.format("mild steel")),
    "L07": (_BLOCK.format("nylon 6/6 plastic"), _PLATE.format("mild steel")),
    "L08": (_BLOCK.format("UHMW polyethylene"), _PLATE.format("mild steel")),
    "L09": (_BLOCK.format("zinc"), _PLATE.format("cast iron")),
    "L10": (_BLOCK.format("copper"), _PLATE.format("cast iron")),
    "L11": (_BLOCK.format("brass"), _PLATE.format("cast iron")),
    "L12": (_BLOCK.format("magnesium"), _PLATE.format("cast iron")),
    "L13": (_BLOCK.format("bronze"), _PLATE.format("cast iron")),
    "L14": (_BLOCK.format("mild-steel"), _PLATE.format("cast iron")),
    "L15": (_BLOCK.format("cast-iron"), _PLATE.format("cast iron")),
    "L16": ("A clean, dry block of oak wood", _PLATE.format("cast iron")),
    "L17": ("A clean, dry block of oak wood, sliding along its grain", "A clean, dry board of oak, grain along the sliding direction"),
    "L18": ("A clean, dry block of oak wood, sliding across its grain", "A clean, dry board of oak, grain along the sliding direction"),
    "L19": ("A clean, dry block of glass with a flat bottom", _PLATE.format("glass")),
    "L20": ("A clean, dry block of graphite (carbon) with a flat bottom", _PLATE.format("glass")),
}

def rows(lookup=DATA/"friction_lookup.csv"):
    out = []
    with open(lookup, newline="") as f:
        for r in csv.DictReader(f):
            if r["pair_id"] not in DESCRIPTIONS:
                continue
            mu = float(r["mu_k_nominal"])
            d_obj, d_surf = DESCRIPTIONS[r["pair_id"]]
            row = dict.fromkeys(FIELDS, "")
            row.update(pair_id=r["pair_id"], object_material=r["object_material"], surface_material=r["surface_material"],
                       mu_k_nominal=r["mu_k_nominal"], s_tab_log=r["s_tab_log"], interval90_low=r["interval90_low"],
                       interval90_high=r["interval90_high"],
                       sources=f"friction_lookup.csv nominal used as simulator truth (lookup tier {r['tier']}; {r['sources']}); the lookup baseline holds this value",
                       in_sim_range=int(0.10 <= mu <= 0.50), description_object=d_obj, description_surface=d_surf)
            out.append(row)
    missing = set(DESCRIPTIONS) - {r["pair_id"] for r in out}
    if missing:
        raise KeyError(f"not in the lookup table: {sorted(missing)}")
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(DATA/"ground_truth_handbook.csv"))
    a = ap.parse_args()
    out = rows()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader(); w.writerows(out)
    print(f"wrote {a.out}: {len(out)} pairs, friction {min(float(r['mu_k_nominal']) for r in out):g}-{max(float(r['mu_k_nominal']) for r in out):g}")

if __name__ == "__main__":
    main()
