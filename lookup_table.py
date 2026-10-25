"""Builds the friction lookup table (CSV + LaTeX rows) from one data structure.
Run: python3 build_table.py  -> writes friction_lookup.csv and rows_*.tex
"""
import csv, math

# Source keys (numbered in the document's reference list, in this order)
SRC = {
 "nasa":   1,  # NASA RP-1228 Table IV (dry sliding), via engineeringlibrary.org
 "elib":   2,  # engineeringlibrary.org reproduction of RP-1228 Table IV
 "etb":    3,  # Engineering ToolBox friction table (cross-check; uncited compilation)
 "yu":     4,  # Yu et al. 2016, MIT pushing dataset (measured dynamic, stainless steel on surface)
 "delrin": 5,  # Delrin 150 datasheet, ASTM D3702, 40 psi, 50 fpm
 "tse":    6,  # TSE Industries, Physical Properties of Plastics (dynamic vs steel)
 "mcm":    7,  # McMaster-Carr, More About Plastics (2016)
 "mens":   8,  # Mens 1991, Wear 149, via Curbell white paper
 "villar": 9,  # Villar-Garcia et al. 2020, chestnut timber, direct shear
}

def s_from_factor(f):
    """log-spread s such that the 90% interval is [mu/f, mu*f]."""
    return math.log(f)/1.645

# Each row: id, object (moving body) material, surface material, nominal mu_k,
# source low, source high (None if single value), s_tab (log spread used for sampling),
# source keys, flags/notes.
# Groups: S = mild-steel plate surface, C = cast-iron plate, O = oak board, G = glass plate,
# Y = Yu et al. surfaces (stainless-steel object).
ROWS = [
 # --- surface: mild steel plate (dry, clean) ------------------------------------
 ("L01","hard steel block",        "mild steel plate", 0.42, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value; mild-on-mild listed 0.57, so steel-on-steel spans 0.42-0.57 across grades"),
 ("L02","aluminium block",         "mild steel plate", 0.47, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value; near the 0.50 cap"),
 ("L03","copper block",            "mild steel plate", 0.36, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value"),
 ("L04","brass block",             "mild steel plate", 0.44, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value"),
 ("L05","phenolic laminate block", "mild steel plate", 0.35, None, None, s_from_factor(1.5), ["nasa","elib"],       "'laminated plastic on steel'; handbook single value"),
 ("L06","acetal (POM) block",      "mild steel plate", 0.20, 0.12, 0.45, s_from_factor(1.8), ["delrin","tse","yu","mens"], "sources disagree by 3x: 0.12-0.16 (Yu, sliding block), 0.20 (D3702), 0.15-0.35 (TSE), 0.45 (Mens); nominal = D3702"),
 ("L07","nylon 6/6 block",         "mild steel plate", 0.27, 0.17, 0.57, s_from_factor(1.8), ["mcm","tse","mens"],  "0.25-0.28 (McMaster), 0.17-0.43 (TSE), 0.57 (Mens); nominal = McMaster midpoint"),
 ("L08","UHMW-PE block",           "mild steel plate", 0.14, 0.09, 0.25, s_from_factor(1.6), ["tse","mcm"],         "0.09-0.12 (TSE), 0.12-0.25 (McMaster); low end gives long slides"),
 # --- surface: cast-iron plate ---------------------------------------------------
 ("L09","zinc block",              "cast iron plate",  0.21, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value"),
 ("L10","copper block",            "cast iron plate",  0.29, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value"),
 ("L11","brass block",             "cast iron plate",  0.30, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value"),
 ("L12","magnesium block",         "cast iron plate",  0.25, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value"),
 ("L13","bronze block",            "cast iron plate",  0.22, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value"),
 ("L14","mild steel block",        "cast iron plate",  0.23, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value"),
 ("L15","cast iron block",         "cast iron plate",  0.15, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value; static 1.10, so static/kinetic ratio is extreme"),
 ("L16","oak block",               "cast iron plate",  0.49, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "'cast iron on oak'; at the 0.50 cap; grain direction unspecified"),
 # --- surface: oak board ---------------------------------------------------------
 ("L17","oak block, grain parallel",      "oak board", 0.48, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value; chestnut-on-chestnut measured 0.25-0.39 (Villar-Garcia) shows wood-on-wood spread"),
 ("L18","oak block, grain perpendicular", "oak board", 0.32, None, None, s_from_factor(1.5), ["nasa","elib","etb","villar"], "handbook single value; within the measured chestnut range"),
 # --- surface: glass plate -------------------------------------------------------
 ("L19","glass block",             "glass plate",      0.40, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value"),
 ("L20","graphite block",          "glass plate",      0.18, None, None, s_from_factor(1.5), ["nasa","elib"],       "'carbon on glass'; handbook single value"),
 # --- Yu et al. surfaces, stainless-steel object (measured, dynamic) --------------
 ("L21","stainless steel block",   "ABS sheet",        0.14, 0.13, 0.15, 0.10, ["yu"], "measured; 0.15 new -> 0.13 after 100 passes; std 0.017"),
 ("L22","stainless steel block",   "acetal (Delrin) sheet", 0.14, 0.12, 0.16, 0.10, ["yu"], "measured; 0.16 -> 0.12 with wear; std 0.016"),
 ("L23","stainless steel block",   "plywood sheet",    0.26, 0.24, 0.28, 0.12, ["yu"], "measured; 0.28 -> 0.24 with wear; largest spatial std (0.064)"),
 ("L24","stainless steel block",   "polyurethane sheet", 0.285, 0.28, 0.29, s_from_factor(1.5), ["yu"], "measured; FLAG: speed-dependent, rises to 1.0 at 0.5 m/s; direction ratio 3/2; use only below 0.3 m/s or exclude"),
]

# Candidate entries found but excluded, with reason.
EXCLUDED = [
 ("mild steel on mild steel", 0.57, "nasa", "above 0.50 cap"),
 ("nickel on nickel", 0.53, "nasa", "above 0.50 cap"),
 ("nickel on mild steel", 0.64, "nasa", "above 0.50 cap"),
 ("glass on nickel", 0.56, "nasa", "above 0.50 cap"),
 ("copper on glass", 0.53, "nasa", "above 0.50 cap"),
 ("leather on oak (parallel)", 0.52, "nasa", "above 0.50 cap"),
 ("leather on cast iron", 0.56, "nasa", "above 0.50 cap"),
 ("aluminium on aluminium", 1.4, "nasa", "above cap; galling"),
 ("mild steel on lead", 0.95, "nasa", "above cap"),
 ("rubber on dry concrete / asphalt", "0.5-0.85", "etb", "above cap; range only"),
 ("PTFE on PTFE / on steel", 0.04, "nasa", "below 0.10 floor; slides too long"),
 ("waxed wood on snow, ice on ice", "0.02-0.10", "etb", "below floor; not a table surface"),
 ("lead on cast iron", 0.43, "nasa", "toxic; excluded for realism (value is usable)"),
 ("tin on cast iron", 0.32, "nasa", "exotic; dropped to reach 24 (value is usable)"),
 ("cadmium on mild steel", 0.46, "nasa", "toxic; excluded"),
 ("magnesium on mild steel", 0.42, "nasa", "dropped to reach 24 (value is usable)"),
 ("hard steel on babbitt", "0.33-0.35", "nasa", "bearing alloy; exotic"),
 ("mild steel on phosphor bronze", 0.34, "nasa", "dropped to reach 24 (value is usable)"),
 ("garnet on mild steel", 0.39, "nasa", "abrasive mineral; not a plausible object"),
 ("polystyrene on steel", "0.30-0.35", "etb", "STATIC only in source"),
 ("Plexiglas (acrylic) on steel", "0.40-0.50", "etb", "STATIC only in source"),
 ("nylon on steel", 0.40, "etb", "STATIC only in source"),
 ("polyethylene on steel", 0.20, "etb", "STATIC only in source"),
 ("wood on clean metal", "0.20-0.60", "etb", "STATIC only in source (the range the 8 Oct draft used as kinetic)"),
 ("wood on wood", "0.25-0.50", "etb", "STATIC only in source"),
 ("leather on wood", "0.30-0.40", "etb", "STATIC only in source"),
 ("paper on cast iron", 0.20, "etb", "STATIC only in source"),
 ("ABS (generic)", "0.3-1.0", "mcm", "range too wide to use; static/dynamic unspecified"),
 ("acrylic, cast (generic)", "0.04-0.5", "mcm", "range too wide to use; static/dynamic unspecified"),
 ("HDPE (generic)", "0.22-0.62", "mcm", "range too wide; static/dynamic unspecified"),
 ("PLA (3D printed) on steel", "~0.30 static", "stoimenov", "static only, derived from reported forces; no kinetic value"),
 ("paper / board on paper", "0.2-0.9", "silveira", "paper-on-paper only; kinetic only in figures"),
]

# Planned primary (measured) set and the lookup entry each would map to.
PRIMARY_OBJECTS = ["PLA (3D printed)", "ABS (3D printed)", "felt pads", "rubber feet", "cardboard", "acrylic", "oak", "ceramic tile (glazed)", "steel washer stack"]
PRIMARY_SURFACES = ["laminate (melamine) desk", "plywood", "painted MDF"]
# mapping: (object, surface) -> (entry or None, note)
def lookup_for(obj, surf):
    if obj.startswith("steel") and surf == "plywood":
        return ("L23 (Yu: steel on plywood 0.24-0.28)", "direct match")
    if obj == "oak" and surf == "plywood":
        return ("L17/L18 class: wood on wood 0.32-0.48", "class match (species and grain differ)")
    if obj.startswith("ABS") and surf == "plywood":
        return (None, "no kinetic entry for ABS on wood; Yu gives steel on ABS only")
    if obj == "acrylic":
        return (None, "acrylic on steel is static-only in sources; nothing on wood or laminate")
    if obj.startswith("PLA"):
        return (None, "no kinetic value found for PLA on anything")
    if obj == "felt pads":
        return (None, "no entry for felt in any source checked")
    if obj == "rubber feet":
        return (None, "rubber entries are on concrete/asphalt and above the 0.50 cap")
    if obj == "cardboard":
        return (None, "paper/board sources are paper-on-paper, static or figure-only")
    if obj.startswith("ceramic"):
        return (None, "no entry; glass-on-glass 0.40 is the nearest and is a weak proxy")
    if obj.startswith("steel"):
        return (None, "steel on laminate / painted MDF not in any source")
    if obj == "oak":
        return (None, "wood on laminate / paint not in any source; wood on wood is a weak proxy")
    if obj.startswith("ABS"):
        return (None, "no entry")
    return (None, "no entry")

def interval(mu, s):
    return mu*math.exp(-1.645*s), mu*math.exp(1.645*s)

def main():
    with open("friction_lookup.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["pair_id","object_material","surface_material","mu_k_nominal","source_low","source_high","s_tab_log","interval90_low","interval90_high","sources","notes"])
        for r in ROWS:
            pid,obj,surf,mu,lo,hi,s,src,note=r
            il,ih=interval(mu,s)
            w.writerow([pid,obj,surf,f"{mu:.3f}", "" if lo is None else lo, "" if hi is None else hi, f"{s:.3f}", f"{il:.3f}", f"{ih:.3f}", ";".join(src), note])
    # LaTeX rows for the main table
    with open("rows_main.tex","w") as f:
        for r in ROWS:
            pid,obj,surf,mu,lo,hi,s,src,note=r
            il,ih=interval(mu,s)
            rng = "single value" if lo is None else f"{lo:.2f}--{hi:.2f}"
            cites = ",".join(f"{SRC[k]}" for k in src)
            flag = "speed-dep." if "FLAG" in note else ("3 sources" if "disagree" in note else ("measured" if "measured" in note else ""))
            f.write(f"{pid} & {obj} & {surf} & {mu:.2f} & {rng} & {il:.2f}--{ih:.2f} & {s:.2f} & [{cites}] & {flag}\\\\\n")
    # LaTeX rows for excluded
    with open("rows_excluded.tex","w") as f:
        for name,val,src,reason in EXCLUDED:
            c = SRC.get(src, {"stoimenov":10,"silveira":11}.get(src,"?"))
            f.write(f"{name} & {val} & [{c}] & {reason}\\\\\n")
    # LaTeX rows for primary mapping
    with open("rows_primary.tex","w") as f:
        n_hit=0; n=0
        for obj in PRIMARY_OBJECTS:
            cells=[]
            for surf in PRIMARY_SURFACES:
                e,note = lookup_for(obj,surf); n+=1
                if e: n_hit+=1; cells.append(e)
                else: cells.append("---")
            f.write(f"{obj} & " + " & ".join(cells) + "\\\\\n")
        print("primary pairs with a lookup entry:", n_hit, "of", n)
    # summary stats
    mus=[r[3] for r in ROWS]
    print("n rows", len(ROWS), "mu range", min(mus), max(mus))
    import statistics
    print("surfaces:", sorted(set(r[2] for r in ROWS)))
    # slide lengths at median/max release speed
    for mu in (0.14, 0.26, 0.49):
        for v in (0.43, 0.8, 1.1):
            print(f"mu={mu} v={v} d={v*v/(2*9.81*mu):.3f} m")

if __name__=="__main__":
    main()
