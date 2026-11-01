#!/usr/bin/env python3
"""
drake_config.py -- one Drake contact configuration, loaded everywhere (sim_config.yaml; spec Sec. 5.1).

* `load_config()` reads sim_config.yaml; `contact_from_config()` turns it into a scene.Contact, so every
  E0 script and the evaluation harness build the same plant.
* The object is compliant, table and paddle rigid: one compliant geometry per contact pair.
* Experiments use the strict `hydroelastic` contact model, so a rigid-rigid pair THROWS instead of
  silently falling back to point contact.  If arm links ever throw, give them hydroelastic properties
  or filter their collisions; do not switch back to `hydroelastic_with_fallback`.
* The Hunt-Crossley dissipation is ignored under SAP, and `relaxation_time` is ignored under Lagged and
  Similar (Drake hydroelastic user guide); `describe()` logs which one is active.
* `assert_hydro_only()` fails if any point-pair contact involves the paddle, the object or the table
  (E0-2, and a runtime check for every rollout).
"""
import os
import yaml
from scene import Contact

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_config.yaml")
HYDRO_KEYS = {"time_step": "time_step", "discrete_contact_approximation": "approx", "contact_model": "contact_model",
              "hydroelastic_modulus": "modulus", "dissipation": "dissipation", "relaxation_time": "relaxation",
              "margin": "margin", "resolution_hint": "resolution"}

def load_config(path=CONFIG_PATH):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k in HYDRO_KEYS:
        if k not in cfg:
            raise KeyError(f"{path} lacks '{k}'")
    return cfg

def contact_from_config(cfg=None, **overrides):
    """scene.Contact from the config dict (default: sim_config.yaml), with keyword overrides for sweeps."""
    cfg = cfg if cfg is not None else load_config()
    kw = {HYDRO_KEYS[k]: cfg[k] for k in HYDRO_KEYS}
    kw["time_step"], kw["modulus"] = float(kw["time_step"]), float(kw["modulus"])
    kw["dissipation"] = float(kw["dissipation"])
    kw["relaxation"] = None if kw["relaxation"] is None else float(kw["relaxation"])
    kw["margin"] = None if kw["margin"] in (None, 0, 0.0) else float(kw["margin"])
    kw["resolution"] = float(kw["resolution"])
    kw.update(overrides)
    return Contact(**kw)

def active_dissipation(approx):
    """Which damping parameter the discrete approximation reads: Hunt-Crossley `dissipation` for lagged and
    similar, `relaxation_time` for sap."""
    return "relaxation_time" if approx == "sap" else "dissipation"

def describe(contact):
    """One line for logs: the settings and which dissipation parameter is active (and which is ignored)."""
    active = active_dissipation(contact.approx)
    damp = (f"relaxation_time={'default' if contact.relaxation is None else contact.relaxation} s (dissipation ignored)"
            if active == "relaxation_time" else f"dissipation={contact.dissipation} s/m (relaxation_time ignored)")
    return (f"contact: {contact.contact_model}, {contact.approx}, dt={contact.time_step*1e3:g} ms, "
            f"E={contact.modulus:g} Pa, {damp}, margin={'default' if contact.margin is None else contact.margin} m, "
            f"resolution={contact.resolution} m")

# ----------------------------------------------------------------------------- contact-result checks

def contact_pairs(results, plant, inspector=None):
    """[(kind, nameA, nameB)] for every contact in a ContactResults: kind is 'point' or 'hydro'.  Names are
    body names for point pairs (strips sit on 'world') and geometry names for hydroelastic contacts when an
    inspector (scene_graph.model_inspector() or query_object.inspector()) is given, else 'hydro'."""
    out = []
    for i in range(results.num_point_pair_contacts()):
        c = results.point_pair_contact_info(i)
        out.append(("point", plant.get_body(c.bodyA_index()).name(), plant.get_body(c.bodyB_index()).name()))
    for i in range(results.num_hydroelastic_contacts()):
        s = results.hydroelastic_contact_info(i).contact_surface()
        if inspector is not None:
            out.append(("hydro", inspector.GetName(s.id_M()), inspector.GetName(s.id_N())))
        else:
            out.append(("hydro", "?", "?"))
    return out

def count_contacts(results):
    return results.num_point_pair_contacts(), results.num_hydroelastic_contacts()

def assert_hydro_only(results, plant, bodies=("paddle", "object", "world")):
    """Raise AssertionError if any point-pair contact involves the paddle, the object or the table (world)."""
    bad = [p for p in contact_pairs(results, plant) if p[0] == "point" and (p[1] in bodies or p[2] in bodies)]
    if bad:
        raise AssertionError("point-pair (non-hydroelastic) contacts: " + ", ".join(f"{a}-{b}" for _, a, b in bad))

if __name__ == "__main__":
    c = contact_from_config()
    print(describe(c))
    print("sap variant:", describe(contact_from_config(approx="sap", relaxation=0.01)))
