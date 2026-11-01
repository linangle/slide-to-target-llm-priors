"""Builds friction lookup table v2 (baseline C2) from one data structure.
Run: python3 lookup_table.py

Writes friction_lookup.csv (v1's 11 columns unchanged, then role, tier, static_upper,
source_speed_mps, source_method, verified) and the LaTeX row files that
friction_lookup_table.tex \\input{}s:
  rows_main.tex        Table 1: L01-L20 handbook rows + Y01-Y04 with independent values
  rows_static.tex      Table 2: static-only / unlabelled / near-miss values (not entries)
  rows_rig.tex         Table 3: rig pairs M01-M27 and V01-V03
  rows_primary.tex     Table 4: object x surface coverage grid built from the M rows
  rows_yu_compare.tex  Yu truth vs independent lookup (truth read from ground_truth_published.csv)
  rows_excluded.tex    other excluded candidates (above cap, toxic, exotic, too wide)
  rows_refs.tex        numbered reference list
ground_truth_published.csv (Yu et al., simulator truth) is read only for the comparison
table and the ID check. Nothing from it enters a lookup row or source list (rule 4).
"""
import csv, math, re, os
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

# ----------------------------------------------------------------------------- sources
# Numbered in this order for the document's reference list. CSV `sources` uses the keys.
# Lines: author(s), title where known, venue, year, URL. Titles marked [title: confirm]
# were not recorded in the spec and must be checked against the PDF (Section 7.5).
SOURCES = [
 # v1 keys ---------------------------------------------------------------------------
 ("nasa",   r"NASA RP-1228, R. T. Barrett, \emph{Fastener Design Manual}, 1990, Table IV (coefficients of friction, dry sliding). \url{https://ntrs.nasa.gov/citations/19900009424}"),
 ("elib",   r"engineeringlibrary.org, reproduction of NASA RP-1228 Table IV (used to read the table; same numbers). \url{https://engineeringlibrary.org/}"),
 ("etb",    r"The Engineering ToolBox, \emph{Friction and Friction Coefficients} (uncited compilation; cross-check only, rule 5). \url{https://www.engineeringtoolbox.com/friction-coefficients-d_778.html}"),
 ("delrin", r"DuPont, \emph{Delrin 150 acetal homopolymer} datasheet: coefficient of friction by ASTM D3702 (thrust washer, 40 psi, 50 fpm $\approx$ 0.25 m/s)."),
 ("tse",    r"TSE Industries, \emph{Physical Properties of Plastics} (table of dynamic coefficient of friction against steel)."),
 ("mcm",    r"McMaster-Carr, \emph{More About Plastics}, 2016 (friction ranges, static/dynamic not always stated)."),
 ("mens",   r"J. W. M. Mens and A. W. J. de Gee, \emph{Friction and wear behaviour of 18 polymers in contact with steel in environments of air and water}, Wear 149 (1991) 255--268; values via a Curbell Plastics white paper."),
 ("villar", r"J. R. Villar-Garc\'{\i}a et al., chestnut timber friction by direct shear, Int. Agrophysics 34, 2020 [title: confirm]."),
 ("stoimenov", r"N. Stoimenov et al., friction of 3D-printed PLA on steel; static value derived from reported forces [full citation: confirm]. Excluded-list only."),
 ("silveira",  r"Silveira et al., paper and board friction (paper on paper; kinetic only in figures) [full citation: confirm]. Excluded-list only."),
 # Section 12 keys (lookup v2) --------------------------------------------------------
 ("BO",  r"Bories Otondo, Montes de Oca and Piovan, 3D-printed PLA friction by inclined-plane launch, AJEA (UTN) 2024 [title: confirm]. \url{https://rtyc.utn.edu.ar/index.php/ajea/article/download/1851/1685/8895}"),
 ("MA",  r"Mahmood et al., tribology of 3D-printed ABS against a stainless hemisphere, Tribology Online 18(6), 2023 [title: confirm]; preprint \url{https://www.researchsquare.com/article/rs-2281753/v1}"),
 ("MI",  r"Mileti\'c et al., tribological behaviour of 3D-printed ABS on steel, SERBIATRIB '25 [title: confirm]. \url{https://scidar.kg.ac.rs/bitstream/123456789/22509/1/184.pdf}"),
 ("DJ",  r"Djurovi\'c et al., \emph{Tribological Behaviour and Surface Roughness Quality of 3D Printed ABS Material}, SERBIATRIB '23. \url{https://scidar.kg.ac.rs/bitstream/123456789/21541/1/Tribological%20Behaviour%20and%20Surface%20Roughness%20Quality%20of%203D%20Printed%20ABS%20Material.pdf}"),
 ("DU",  r"DuPont, \emph{Delrin low wear / low friction grades}, brochure L-12259 (block-on-ring data against steel). \url{https://www.emcoplastics.com/assets/pdf/delrin/Delrin%20Low%20Wear%20Low%20Friction%20Brochure%20Dupont.pdf}"),
 ("MZ",  r"Mart\'{\i}nez et al., TPU--steel long-stroke sliding tests, LubMat '12 [title: confirm]. \url{https://core.ac.uk/download/16754648.pdf}"),
 ("KU",  r"Kukla, Wargu\l{}a and Biszczanik, \emph{Determining the Coefficient of Friction of Wood-Based Materials for Furniture Panels in the Aspect of Modelling Their Shredding Process}, Wood Research 66(5), 2021. \url{https://www.woodresearch.sk/cms/wp-content/uploads/2021/10/Determining-The-Coefficient-Of-Friction-Of-Wood-Based-Materals-For-Furniture-Panels-In-The-Aspect-Of-Modelling-Their-Shredding-Process.pdf}"),
 ("ME",  r"Meng, Hirai and Koizumi, friction between timber and steel (Table 3), Mokuzai Gakkaishi 54(5), 2008 [title: confirm]. \url{https://eprints.lib.hokudai.ac.jp/repo/huscap/all/68014/54_281.pdf}"),
 ("VI",  r"Villar-Garc\'{\i}a et al., timber friction coefficients, Forests 13(7), 1078, 2022 [title: confirm]. \url{https://www.mdpi.com/1999-4907/13/7/1078}"),
 ("WH",  r"S. V. Glass and S. L. Zelinka, \emph{Moisture Relations and Physical Properties of Wood}, Ch. 4 in \emph{Wood Handbook: Wood as an Engineering Material}, FPL-GTR-190, 2010, p. 4-17 (friction). \url{https://archive.org/details/wood-as-an-engineering-material}"),
 ("KO",  r"Kollmann (1959) and Morin values for wood-on-wood friction, as tabulated secondhand in Aira et al., Mater. Construcc. 64(315), 2014 (\url{https://materconstrucc.revistas.csic.es/index.php/materconstrucc/article/download/1482/1746?inline=1}) and Villar-Garc\'{\i}a et al., Int. Agrophysics 34, 2020. Kollmann not opened."),
 ("VA",  r"Valiant, friction of shoe-sole rubber on sports floors (varnished maple), ISBS 1986 [title: confirm; poor scan, speed unreadable]. \url{https://ojs.ub.uni-konstanz.de/cpa/article/view/1497/1347}"),
 ("RO",  r"F. L. Roth, R. L. Driscoll and W. L. Holt, \emph{Frictional properties of rubber}, J. Res. NBS 28 (1942) 439. \url{https://nvlpubs.nist.gov/nistpubs/jres/28/jresv28n4p439_A1b.pdf}"),
 ("GDV", r"GDV (German Insurance Association), \emph{Container Handbook}, Sec. 4.4.5, friction tables 1c--1d. \url{https://www.containerhandbuch.de/chb_e/stra/stra_04_04_05.html}"),
 ("YA",  r"Yang, Luo and Cai, friction of corrugated boxes on wood, WCMNM 2017 [title: confirm]. \url{https://core.ac.uk/download/82919615.pdf}"),
 ("PA",  r"Papalou, tilt-test friction coefficients of common materials, Applied Mechanics 5(4), 48, 2024 [title: confirm]. \url{https://www.mdpi.com/2673-3161/5/4/48}"),
 ("CTU", r"IMO/ILO/UNECE \emph{CTU Code}, Quick Lashing Guide (IMO MSC.1/Circ.1498) and the Cordstrap CTU Code guide: static-derived design friction factors (not kinetic measurements)."),
 ("PLX", r"R\"{o}hm, PLEXIGLAS (cast acrylic) product data: friction values 0.8 and 0.5 quoted without counterface or static/kinetic label [datasheet and URL: locate before citing]. Table 2 only."),
]
SRC = {k: i+1 for i, (k, _) in enumerate(SOURCES)}   # key -> reference number
BIB = dict(SOURCES)
# Not a lookup source (rule 4): Yu et al. 2016 sets simulator friction for Y01-Y04.
# The document cites it in its own text; the comparison table reads its values from the truth file.
YU_TRUTH_FILE = "ground_truth_published.csv"

# ----------------------------------------------------------------------------- spreads
TIER_F = {"A": 1.5, "B": 2.0, "C": 2.5}      # rule 3: 90% interval mu/f .. mu*f

def s_from_factor(f):
    """log-spread s such that the 90% interval is [mu/f, mu*f]."""
    return math.log(f)/1.645

def interval(mu, s):
    return mu*math.exp(-1.645*s), mu*math.exp(1.645*s)

# ----------------------------------------------------------------------------- data
# PLACEHOLDER DECISIONS (Section 10): the team has not decided these yet. The spec's
# recommendations are used so the files build; change here once decided.
#   decision 2 (V03): "crossed" 0.19 (90% 0.13-0.29, static 0.43) is the nominal;
#                     "perpendicular" 0.33 (0.22-0.50, static 0.54) is kept in notes.
#   decision 3 (tiers): spread factors 1.5 / 2 / 2.5 for tiers A / B / C are accepted.
#   decision 5 (L06): nominal stays 0.20 (ASTM D3702); [DU] 0.30-0.33 at 0.5 m/s is
#                     added as a further source, not as the nominal.
PLACEHOLDERS = [
 ("2", "V03 nominal = 'crossed' 0.19 (alt. 'perpendicular' 0.33 in notes)"),
 ("3", "tier spread factors 1.5 / 2 / 2.5 accepted"),
 ("5", "L06 nominal kept at 0.20 (ASTM D3702); [DU] 0.30-0.33 added as a source"),
]

@dataclass
class Row:
    pid: str
    obj: str
    surf: str
    mu: float            # None when tier == "none"
    lo: float            # source low / high (None if a single value)
    hi: float
    s: float             # log spread (None when tier == "none")
    src: list
    note: str
    role: str            # secondary (L01-L20) | yu_pair (Y01-Y04) | rig (M, V)
    tier: str            # A | B | C | none   (for L01-L20 informational: spreads are v1's)
    static_upper: float = None
    speed: str = ""      # test speed in the source, m/s, blank if unstated
    method: str = ""     # test method in the source
    verified: str = "n"  # y once checked against the source PDF (Section 7.5)
    flag: str = ""       # short flag printed in Table 1
    alias: str = ""      # v1 ID for the Yu pairs (shown as "Y01 (L21)")
    match: str = ""      # "source and match" text for the Yu comparison table

def L(pid, obj, surf, mu, lo, hi, s, src, note, tier="A", **kw):
    return Row(pid, obj, surf, mu, lo, hi, s, src, note, "secondary", tier, **kw)

# --- L01-L20: handbook rows from v1, unchanged values and spreads ------------------
# Groups: mild-steel plate, cast-iron plate, oak board, glass plate surfaces.
# Nothing here may come from data used to set simulator friction (our measurements, Yu et al. 2016).
# Tier letter is informational for these rows: all keep v1's spreads (f = 1.5, wider where
# sources disagree). Simulator truth for L01-L20 is sampled from this table's own interval,
# so C2 is correct there by construction (memorisation probe, see the document).
ROWS_L = [
 L("L01","hard steel block",        "mild steel plate", 0.42, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value; mild-on-mild listed 0.57, so steel-on-steel spans 0.42-0.57 across grades", method="handbook table"),
 L("L02","aluminium block",         "mild steel plate", 0.47, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value; near the 0.50 cap", method="handbook table", flag="near 0.50 cap"),
 L("L03","copper block",            "mild steel plate", 0.36, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value", method="handbook table"),
 L("L04","brass block",             "mild steel plate", 0.44, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value", method="handbook table"),
 L("L05","phenolic laminate block", "mild steel plate", 0.35, None, None, s_from_factor(1.5), ["nasa","elib"],       "'laminated plastic on steel'; handbook single value", method="handbook table"),
 L("L06","acetal (POM) block",      "mild steel plate", 0.20, 0.15, 0.45, s_from_factor(1.8), ["delrin","tse","mens","DU"], "sources disagree by 3x: 0.20 (D3702), 0.15-0.35 (TSE), 0.45 (Mens), 0.30-0.33 at 0.5 m/s [DU]; nominal = D3702 (PLACEHOLDER, Section 10 decision 5)", speed="0.25 (D3702); 0.5 (DU)", method="thrust washer ASTM D3702; block-on-ring [DU]", flag="4 sources"),
 L("L07","nylon 6/6 block",         "mild steel plate", 0.27, 0.17, 0.57, s_from_factor(1.8), ["mcm","tse","mens"],  "0.25-0.28 (McMaster), 0.17-0.43 (TSE), 0.57 (Mens); nominal = McMaster midpoint", method="datasheet / handbook", flag="3 sources"),
 L("L08","UHMW-PE block",           "mild steel plate", 0.14, 0.09, 0.25, s_from_factor(1.6), ["tse","mcm"],         "0.09-0.12 (TSE), 0.12-0.25 (McMaster); low end gives long slides", method="datasheet / handbook"),
 L("L09","zinc block",              "cast iron plate",  0.21, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value", method="handbook table"),
 L("L10","copper block",            "cast iron plate",  0.29, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value", method="handbook table"),
 L("L11","brass block",             "cast iron plate",  0.30, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value", method="handbook table"),
 L("L12","magnesium block",         "cast iron plate",  0.25, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value", method="handbook table"),
 L("L13","bronze block",            "cast iron plate",  0.22, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value", method="handbook table"),
 L("L14","mild steel block",        "cast iron plate",  0.23, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value", method="handbook table"),
 L("L15","cast iron block",         "cast iron plate",  0.15, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value; static 1.10, so static/kinetic ratio is extreme", static_upper=1.10, method="handbook table", flag="static 1.10"),
 L("L16","oak block",               "cast iron plate",  0.49, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "'cast iron on oak', reversed geometry; at the 0.50 cap; grain direction unspecified", tier="C", method="handbook table", flag="near 0.50 cap"),
 L("L17","oak block, grain parallel",      "oak board", 0.48, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value; chestnut-on-chestnut measured 0.25-0.39 (Villar-Garcia) shows wood-on-wood spread; same figure as Kollmann's oak on oak (V02)", method="handbook table"),
 L("L18","oak block, grain perpendicular", "oak board", 0.32, None, None, s_from_factor(1.5), ["nasa","elib","etb","villar"], "handbook single value; within the measured chestnut range", method="handbook table; direct shear [villar]"),
 L("L19","glass block",             "glass plate",      0.40, None, None, s_from_factor(1.5), ["nasa","elib","etb"], "handbook single value", method="handbook table"),
 L("L20","graphite block",          "glass plate",      0.18, None, None, s_from_factor(1.5), ["nasa","elib"],       "'carbon on glass'; handbook single value", method="handbook table"),
]

# --- Y01-Y04: Yu et al.'s pairs with INDEPENDENT lookup values (Section 7.2) ----------
# IDs match the truth file ground_truth_published.csv. Material strings match it too.
# The Yu figures themselves are not stored here (rule 4); the comparison table reads them
# from the truth file at run time.
def Y(pid, alias, obj, surf, mu, tier, src, note, match, **kw):
    return Row(pid, obj, surf, mu, kw.pop("lo", None), kw.pop("hi", None), s_from_factor(TIER_F[tier]),
               src, f"{alias} in v1 (Yu value removed: simulator truth, not a lookup source); " + note,
               "yu_pair", tier, alias=alias, match=match, flag="replaces Yu value", **kw)

ROWS_Y = [
 Y("Y01","L21","stainless steel block","ABS sheet",             0.24, "C", ["MA"],
   "printed ABS under a stainless hemisphere, 0.05 m/s [MA]; printed, not extruded sheet, so tier C",
   "[MA]: printed ABS under a stainless hemisphere, 0.05 m/s", speed="0.05", method="hemisphere-on-flat"),
 Y("Y02","L22","stainless steel block","acetal (Delrin) sheet", 0.31, "A", ["DU"],
   "Delrin block on steel ring, 0.5 m/s, 10 N [DU]; exact materials at a matching speed, but a loaded ring test, not a lightly loaded flat block (ASTM G115: friction belongs to the whole test system)",
   "[DU]: Delrin block on steel ring, 0.5 m/s, 10 N", speed="0.5", method="block-on-ring, 10 N"),
 Y("Y03","L23","stainless steel block","plywood sheet",         0.22, "C", ["ME","VI"],
   "smooth steel and timber, reversed roles [ME], [VI]; Meng 2008 Table 3 row assignment garbled in extraction, verify",
   "[ME], [VI]: smooth steel and timber, reversed roles", method="timber on steel plate, reversed roles"),
 Y("Y04","L24","stainless steel block","polyurethane sheet",    0.61, "B", ["MZ"],
   "TPU 95A on brushed steel, flat on flat, 0.32 m/s [MZ]; above the simulated range (0.10-0.50); rubber-like friction rises with speed [RO]",
   "[MZ]: TPU 95A on brushed steel, flat on flat, 0.32 m/s", speed="0.32", method="flat-on-flat long stroke"),
]

# --- rig pairs M01-M27 and V01-V03 (Section 7.3) -------------------------------------
# IDs as in the measured file: M = 9(surface - 1) + object, so M18 is O9 on S2.
# Material strings match ground_truth_placeholder.csv.
OBJECTS = {"1":"PLA (3D printed)", "2":"ABS sheet (extruded)", "3":"furniture felt", "4":"neoprene rubber 60A",
           "5":"corrugated cardboard", "6":"cast acrylic", "7":"oak, grain along travel",
           "8":"glazed ceramic tile", "9":"stainless steel 304", "7b":"oak, grain across travel"}
SURFACES = {"1":"white melamine shelf board", "2":"birch plywood, sanded", "3":"painted MDF (satin)",
            "4":"oak board, grain along travel", "5":"ABS sheet, smooth face"}

def rig(pid, mu, tier, src, note, static=None, speed="", method="", lo=None, hi=None, obj=None, surf=None):
    if pid.startswith("M"):
        n = int(pid[1:]); o, su = str((n-1) % 9 + 1), str((n-1)//9 + 1)
        obj, surf = obj or o, surf or su
    s = s_from_factor(TIER_F[tier]) if tier in TIER_F else None
    if tier not in TIER_F:
        assert mu is None and lo is None and hi is None, pid
    return Row(pid, OBJECTS[obj], SURFACES[surf], mu, lo, hi, s, src, note, "rig", tier,
               static_upper=static, speed=speed, method=method)

ABS_NOTE = "Printed ABS on steel, 0.05-0.75 m/s (sheet ABS on wood/laminate not found: different counterface class)"
ROWS_M = [
 # --- S1 white melamine shelf board -----------------------------------------------
 rig("M01", 0.29, "C", ["BO"],           "PLA on acrylic, inclined-plane launch", method="inclined plane"),
 rig("M02", 0.30, "C", ["MI","DJ","MA"], ABS_NOTE, speed="0.05-0.75", method="tribometer, printed ABS on steel"),
 rig("M03", None, "none", ["PA"],        "Felt and ceramic on a melamine base; geometry unclear; static only", static=0.41, method="tilt test (static)"),
 rig("M04", 0.83, "C", ["VA"],           "Rubber sole on varnished maple; rubber friction rises with speed [RO]; above the simulated range", method="sole on floor, sled type; speed unreadable in scan"),
 rig("M05", 0.20, "C", ["GDV"],          "Cardboard on resin-coated board, sliding", method="handbook sliding value"),
 rig("M06", None, "none", [],            "No kinetic source"),
 rig("M07", 0.39, "C", ["WH"],           "Dry wood on hard smooth surfaces, 0.3-0.5 (geometric mean)", lo=0.30, hi=0.50, method="handbook range"),
 rig("M08", None, "none", ["PA"],        "Ceramic on melamine base, tilt test; static only", static=0.28, method="tilt test (static)"),
 rig("M09", 0.25, "B", ["KU"],           "Wear-plate steel on decor-faced chipboard, 22 mm/s", speed="0.022", method="sled on panel"),
 # --- S2 birch plywood, sanded ------------------------------------------------------
 rig("M10", 0.33, "B", ["BO"],           "PLA on fine-sanded wood", method="inclined plane"),
 rig("M11", 0.30, "C", ["MI","DJ","MA"], "As M02", speed="0.05-0.75", method="tribometer, printed ABS on steel"),
 rig("M12", None, "none", [],            "No source"),
 rig("M13", 0.83, "C", ["VA"],           "As M04", method="sole on floor, sled type; speed unreadable in scan"),
 rig("M14", 0.21, "B", ["YA","GDV","CTU"], "Corrugated box on smooth pine; static upper from CTU design value", static=0.50, method="box slid on pine board"),
 rig("M15", None, "none", [],            "No kinetic source"),
 rig("M16", 0.34, "C", ["ME","KO"],      "Geometric mean of timber on plywood (0.24) and oak on oak (0.48)", lo=0.24, hi=0.48, method="geometric mean of two sources"),
 rig("M17", None, "none", [],            "No source"),
 rig("M18", 0.22, "C", ["ME","VI","CTU"], "Smooth steel and timber, reversed roles; static upper from CTU design value", static=0.45, method="timber on steel plate, reversed roles"),
 # --- S3 painted MDF (satin) --------------------------------------------------------
 rig("M19", 0.29, "C", ["BO"],           "PLA on acrylic", method="inclined plane"),
 rig("M20", 0.30, "C", ["MI","DJ","MA"], "As M02", speed="0.05-0.75", method="tribometer, printed ABS on steel"),
 rig("M21", None, "none", [],            "No source"),
 rig("M22", 0.83, "C", ["VA"],           "Varnished surface; as M04", method="sole on floor, sled type; speed unreadable in scan"),
 rig("M23", 0.20, "C", ["GDV"],          "Coated board; as M05", method="handbook sliding value"),
 rig("M24", None, "none", [],            "No kinetic source"),
 rig("M25", 0.39, "C", ["WH"],           "As M07", lo=0.30, hi=0.50, method="handbook range"),
 rig("M26", None, "none", [],            "No source"),
 rig("M27", None, "none", [],            "Bare-MDF value (0.68 [KU]) excluded: paint changes the surface"),
 # --- validation pairs --------------------------------------------------------------
 rig("V01", 0.24, "C", ["MA"],           "Same source as Y01 (L21)", speed="0.05", method="hemisphere-on-flat", obj="9", surf="5"),
 rig("V02", 0.48, "A", ["KO"],           "Morin and Kollmann values, secondhand; same figure as handbook L17", static=0.62, method="tabulated (secondhand)", obj="7", surf="4"),
 rig("V03", 0.19, "A", ["KO"],           "PLACEHOLDER (Section 10 decision 2): 'crossed' 0.19 (90% 0.13-0.29, static 0.43) used; 'perpendicular' 0.33 (90% 0.22-0.50, static 0.54) is the alternative; team decides", static=0.43, lo=0.19, hi=0.33, method="tabulated (secondhand)", obj="7b", surf="4"),
]

ROWS = ROWS_L + ROWS_Y + ROWS_M

# --- static-only, unlabelled and near-miss values that are NOT entries (Table 2) -----
# (pair, value, source keys, type, note)
STATIC = [
 ("furniture felt on melamine-faced base",          "0.41",        ["PA"],  "static",     "tilt test; geometry unclear; recorded as M03 static upper"),
 ("glazed ceramic on melamine-faced base",          "0.28",        ["PA"],  "static",     "tilt test; recorded as M08 static upper"),
 ("cardboard (untreated) on wooden pallet",         "0.50",        ["CTU"], "static-derived design value", "lashing-guide factor; recorded as M14 static upper"),
 ("unpainted rough steel on sawn wood",             "0.45",        ["CTU"], "static-derived design value", "lashing-guide factor; recorded as M18 static upper"),
 ("oak on oak, grain along (static)",               "0.62",        ["KO"],  "static",     "secondhand; recorded as V02 static upper"),
 ("oak across on oak along (static)",               "0.54 perp. / 0.43 crossed", ["KO"], "static", "secondhand; V03 static upper follows the Section 10 decision"),
 ("PLEXIGLAS (cast acrylic), datasheet",            "0.8 and 0.5", ["PLX"], "unlabelled", "counterface and static/kinetic not stated; cannot be used (M06, M15, M24 stay none)"),
 ("wear-plate steel on bare (unpainted) MDF",       "0.68",        ["KU"],  "kinetic, excluded", "bare MDF; paint changes the surface, so not used for M27"),
]

# Candidate entries found but excluded, with reason. Entries whose reason mentions STATIC
# are printed in Table 2 with the list above; the rest go to the excluded table.
# The four Yu et al. pairs are no longer listed here: they are Y01-Y04 with independent values.
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
 ("PLA (3D printed) on steel", "approx. 0.30 static", "stoimenov", "STATIC only, derived from reported forces; no kinetic value"),
 ("paper / board on paper", "0.2-0.9", "silveira", "paper-on-paper only; kinetic only in figures"),
]

# ----------------------------------------------------------------------------- formatting
def fmt(x):
    """Table display: 2 decimals, half-up; 3 decimals below 0.10 or for exact thousandth-halves
    under 0.20 (0.125, 0.165, 0.105), matching the spec tables."""
    if x is None: return ""
    if isinstance(x, str): return x
    d = Decimal(repr(x)).quantize(Decimal("1e-9"))          # kill binary noise before rounding
    if x < 0.0995: return str(d.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)).rstrip("0")
    m = x*1000
    if x < 0.2 and abs(m - round(m)) < 1e-6 and round(m) % 10 == 5:
        return f"{x:.3f}"
    return str(d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def fmt_val(v):
    """Values in the static/excluded lists: floats to 2 decimals, strings as given."""
    return f"{v:.2f}" if isinstance(v, float) else str(v)

def csvnum(x, nd=3):
    return "" if x is None else f"{x:.{nd}f}"

_KEY_RE = re.compile(r"\[(" + "|".join(sorted(SRC, key=len, reverse=True)) + r")\]")

def tex(s):
    """Escape plain text for LaTeX; [KEY] -> [n]; digit-digit ranges -> en dash."""
    if s is None: return ""
    s = str(s)
    s = _KEY_RE.sub(lambda m: f"[{SRC[m.group(1)]}]", s)
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    s = s.replace("->", r"$\to$").replace("<", r"$<$").replace(">", r"$>$")
    s = re.sub(r"(\d)-(\d)", r"\1--\2", s)
    return s

def cites(keys):
    return "[" + ",".join(str(SRC[k]) for k in keys) + "]" if keys else "---"

# ----------------------------------------------------------------------------- outputs
CSV_COLS = ["pair_id","object_material","surface_material","mu_k_nominal","source_low","source_high",
            "s_tab_log","interval90_low","interval90_high","sources","notes",
            "role","tier","static_upper","source_speed_mps","source_method","verified"]

def write_csv(path="friction_lookup.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in ROWS:
            if r.mu is None:
                il = ih = None
            else:
                il, ih = interval(r.mu, r.s)
            w.writerow([r.pid, r.obj, r.surf, csvnum(r.mu), csvnum(r.lo, 2), csvnum(r.hi, 2), csvnum(r.s),
                        csvnum(il), csvnum(ih), ";".join(r.src), r.note,
                        r.role, r.tier, csvnum(r.static_upper, 2), r.speed, r.method, r.verified])

def write_rows_main(path="rows_main.tex"):
    with open(path, "w") as f:
        for r in ROWS_L + ROWS_Y:
            il, ih = interval(r.mu, r.s)
            rng = "single value" if r.lo is None else f"{r.lo:.2f}--{r.hi:.2f}"
            pid = f"{r.pid} ({r.alias})" if r.alias else r.pid
            f.write(f"{pid} & {tex(r.obj)} & {tex(r.surf)} & {fmt(r.mu)} & {rng} & {fmt(il)}--{fmt(ih)} & {r.s:.2f} & {r.tier} & {cites(r.src)} & {tex(r.flag)}\\\\\n")

def write_rows_rig(path="rows_rig.tex"):
    with open(path, "w") as f:
        for r in ROWS_M:
            if r.mu is None:
                mu = iv = "---"
            else:
                il, ih = interval(r.mu, r.s)
                mu, iv = fmt(r.mu), f"{fmt(il)}--{fmt(ih)}"
            st = fmt(r.static_upper) if r.static_upper is not None else ""
            f.write(f"{r.pid} & {tex(r.obj)} & {tex(r.surf)} & {mu} & {r.tier} & {iv} & {st} & {cites(r.src)} & {tex(r.note)}\\\\\n")

def write_rows_static(path="rows_static.tex"):
    with open(path, "w") as f:
        for pair, val, keys, kind, note in STATIC:
            f.write(f"{tex(pair)} & {tex(val)} & {kind} & {cites(keys)} & {tex(note)}\\\\\n")
        for name, val, src, reason in EXCLUDED:
            if "STATIC" in reason:
                f.write(f"{tex(name)} & {tex(fmt_val(val))} & static & {cites([src])} & {tex(reason.replace('STATIC only in source', 'v1 excluded list').replace('STATIC only', 'v1 excluded list'))}\\\\\n")

def write_rows_excluded(path="rows_excluded.tex"):
    with open(path, "w") as f:
        for name, val, src, reason in EXCLUDED:
            if "STATIC" not in reason:
                f.write(f"{tex(name)} & {tex(fmt_val(val))} & {cites([src])} & {tex(reason)}\\\\\n")

def write_rows_primary(path="rows_primary.tex"):
    """Object x surface grid of the M rows: mu (tier), or --- with any static upper."""
    by_pair = {(r.obj, r.surf): r for r in ROWS_M}
    with open(path, "w") as f:
        for o in map(str, range(1, 10)):
            cells = []
            for su in ("1", "2", "3"):
                r = by_pair[(OBJECTS[o], SURFACES[su])]
                if r.mu is None:
                    cells.append("---" + (f" (static {fmt(r.static_upper)})" if r.static_upper else ""))
                else:
                    cells.append(f"{fmt(r.mu)} ({r.tier})")
            f.write(f"O{o} {tex(OBJECTS[o])} & " + " & ".join(cells) + "\\\\\n")

def read_yu_truth(path=YU_TRUTH_FILE):
    """Yu et al. simulator-truth values by pair ID, for the comparison table only."""
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {row["pair_id"]: float(row["mu_k_nominal"]) for row in csv.DictReader(f) if row["pair_id"].startswith("Y")}

def write_rows_yu_compare(truth, path="rows_yu_compare.tex"):
    with open(path, "w") as f:
        for r in ROWS_Y:
            il, ih = interval(r.mu, r.s)
            yu = truth.get(r.pid)
            yu_s = f"{yu:.3g}" if yu is not None else "n/a"
            ratio = f"{r.mu/yu:.2g}$\\times$" if yu else "n/a"
            f.write(f"{r.pid} ({r.alias}) {tex(r.obj)} on {tex(r.surf)} & {yu_s} & {fmt(r.mu)} & {r.tier} & {fmt(il)}--{fmt(ih)} & {ratio} & {tex(r.match)}\\\\\n")

def write_rows_refs(path="rows_refs.tex"):
    with open(path, "w") as f:
        for k, line in SOURCES:
            f.write(f"\\bibitem[{SRC[k]}]{{{k}}} \\textbf{{{tex(k)}}} --- {line}\n")

# ----------------------------------------------------------------------------- checks
def check_consistency():
    """Sanity checks on the data: IDs, materials, intervals, truth-file IDs."""
    ids = [r.pid for r in ROWS]
    assert len(ids) == len(set(ids)), "duplicate pair IDs"
    for r in ROWS:
        for k in r.src:
            assert k in SRC, f"{r.pid}: unknown source key {k}"
        assert "yu" not in [k.lower() for k in r.src], f"{r.pid}: Yu is not a lookup source"
        if r.tier == "none":
            assert r.mu is None and r.s is None, f"{r.pid}: tier none must have no value"
        else:
            assert r.mu is not None and r.s is not None, f"{r.pid}: missing value"
            if r.role != "secondary":
                assert abs(r.s - s_from_factor(TIER_F[r.tier])) < 1e-12, f"{r.pid}: spread does not match tier"
    # material strings against the measured-file placeholder, if present
    if os.path.exists("ground_truth_placeholder.csv"):
        with open("ground_truth_placeholder.csv", newline="") as f:
            gt = {row["pair_id"]: (row["object_material"], row["surface_material"]) for row in csv.DictReader(f)}
        for r in ROWS_M:
            if r.pid in gt and gt[r.pid] != (r.obj, r.surf):
                print(f"WARNING {r.pid}: materials differ from ground_truth_placeholder.csv: {gt[r.pid]} vs {(r.obj, r.surf)}")
    truth = read_yu_truth()
    for r in ROWS_Y:
        if truth and r.pid not in truth:
            print(f"WARNING {r.pid}: not found in {YU_TRUTH_FILE}")
    return truth

def main():
    truth = check_consistency()
    write_csv()
    write_rows_main()
    write_rows_rig()
    write_rows_static()
    write_rows_excluded()
    write_rows_primary()
    write_rows_yu_compare(truth)
    write_rows_refs()
    # re-read the CSV to make sure it parses
    with open("friction_lookup.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [c for c in rows[0].keys()] == CSV_COLS
    # coverage
    m_rows = [r for r in ROWS_M if r.pid.startswith("M")]
    covered = [r for r in m_rows if r.tier != "none"]
    per_tier = {t: sum(1 for r in m_rows if r.tier == t) for t in ("A", "B", "C", "none")}
    print(f"rig coverage: {len(covered)} of {len(m_rows)} M-pairs have a tier other than none; per tier {per_tier}")
    v_tiers = {r.pid: r.tier for r in ROWS_M if r.pid.startswith("V")}
    print("validation pairs:", v_tiers)
    print(f"rows written: {len(rows)} ({len(ROWS_L)} secondary, {len(ROWS_Y)} yu_pair, {len(ROWS_M)} rig); sources: {len(SOURCES)}")
    mus = [r.mu for r in ROWS if r.mu is not None]
    print("mu range over entries", min(mus), max(mus), "; entries outside 0.10-0.50:",
          [r.pid for r in ROWS if r.mu is not None and not 0.10 <= r.mu <= 0.50])
    print("REMINDER: Section 10 decisions are PLACEHOLDERS -- " + "; ".join(f"d{d}: {t}" for d, t in PLACEHOLDERS))
    # slide lengths at median/max release speed, for a feel of the numbers
    for mu in (0.14, 0.26, 0.49):
        for v in (0.43, 0.8, 1.1):
            print(f"mu={mu} v={v} d={v*v/(2*9.81*mu):.3f} m")

if __name__ == "__main__":
    main()
