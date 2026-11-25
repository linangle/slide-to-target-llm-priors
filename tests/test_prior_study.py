"""paired_stats.py, prior_study.py and probes.py: the proposal items that need no new language-model answers."""
import json, math
from types import SimpleNamespace as NS
import numpy as np
import pytest
import llm_prior
import paired_stats as S
import prior_study as PS
import probes
import query_claude as Q
import run_pair_set as R
from belief import logprior, posterior, theta_grid
from paths import DATA

def test_paired_statistics_over_pairs():
    a, b = [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7]
    s = S.paired(a, b)
    assert s["diff"] == -1 and s["lo"] == -1 and s["hi"] == -1 and s["a_wins"] == 6 and s["b_wins"] == 0
    assert abs(s["p"] - 2/2**6) < 1e-12                        # six pairs all one way: the smallest p six pairs can give
    same = S.paired(a, a)
    assert same["diff"] == 0 and same["p"] == 1.0 and same["ties"] == 6
    noisy = S.paired([5, 1, 4, 2, 6, 3, 7, 2], [4, 2, 3, 3, 5, 4, 6, 3])
    assert noisy["lo"] < noisy["diff"] < noisy["hi"] and noisy["a_wins"] + noisy["b_wins"] == 8
    with pytest.raises(ValueError):
        S.paired([1, 2], [1, 2, 3])
    rows = [dict(pair="A", method="x", slides="0", error_x_mm="-4", valid="True"), dict(pair="A", method="x", slides="0", error_x_mm="2", valid="True"),
            dict(pair="A", method="x", slides="0", error_x_mm="99", valid="False"), dict(pair="B", method="x", slides="1", error_x_mm="1", valid="True")]
    assert S.per_pair(rows) == {("x", 0): {"A": 3.0}, ("x", 1): {"B": 1.0}}        # seeds and targets averaged inside a pair; invalid rows left out

def test_robust_mixture_and_interval_score():
    g = theta_grid(2000, math.log(.005), math.log(5.))
    narrow, wide = logprior(g, math.log(.3), .1), logprior(g, math.log(.3), .8)
    mix = PS.mixture(g, narrow, wide)
    assert abs(np.exp(mix).sum()*(g[1] - g[0]) - 1) < 1e-6                         # a density
    far = int(np.argmin(np.abs(g - math.log(.8))))
    assert mix[far] > narrow[far] + 5 and abs(mix[far] - (math.log(1 - PS.ROBUST_WEIGHT) + PS.normalised(g, wide)[far])) < 1e-6   # the wide tail is kept
    post = posterior(g, narrow)
    inside, outside = PS.interval_row(g, post, .3), PS.interval_row(g, post, .6)
    assert inside["inside"] == 1 and abs(inside["interval_score"] - inside["width"]) < 1e-12
    assert outside["inside"] == 0 and abs(outside["interval_score"] - (outside["width"] + 20*(math.log(.6) - post["q95"]))) < 1e-9
    assert outside["log_score"] > inside["log_score"] and abs(inside["width"] - 2*1.645*.1) < 0.01

def test_probe_prompts_are_built_from_descriptions_only():
    some = ["L01", "L16", "Y01"]
    truth = {p: d["mu"] for p, d in R.load_pairs(R.TRUTH, some).items()}
    counts = {kind: len(probes.build(kind, only=some)) for kind in probes.KINDS}
    assert counts == dict(wording_b=3, hidden=3, counterfactual=3*len(probes.COUNTERFACTUALS), recite=3, speed=9)
    for kind in ("wording_b", "hidden", "counterfactual", "recite"):
        for d in probes.build(kind, only=some).values():
            assert f"{truth[d['pair']]:g}" not in d["prompt"] and f"{truth[d['pair']]:.3f}" not in d["prompt"]      # never the friction
    desc = probes.descriptions(only=["L01"])["L01"]
    assert desc["description_object"] not in probes.build("hidden", only=["L01"])["L01/hidden"]["prompt"]          # the material is withheld
    assert probes.HIDDEN_OBJECT in probes.build("hidden", only=["L01"])["L01/hidden"]["prompt"]
    for d in probes.build("counterfactual", only=["L01"]).values():
        assert d["expect"] in ("lower", "higher") and desc["description_object"] in d["prompt"]
    assert probes.build("wording_b", only=["L01"])["L01/wording_b"]["prompt"] != R.load_pairs(R.TRUTH, ["L01"])["L01"]["prompt"]
    assert "0.050 m beyond" in probes.build("speed", only=["L01"])["L01/speed@0.60"]["prompt"]
    with pytest.raises(ValueError):
        probes.build("photo")

def test_probe_answers_are_validated():
    assert probes.parse_recite(json.dumps(dict(recalls_value=True, value=0.42, source="a handbook")))["value"] == 0.42
    assert probes.parse_recite(json.dumps(dict(recalls_value=False, value=None, source="")))["recalls_value"] is False
    for bad in (dict(recalls_value=True, value=None, source="x"), dict(recalls_value=False, value=0.4, source=""), dict(recalls_value=True, value=-1, source="x")):
        with pytest.raises(ValueError):
            probes.parse_recite(json.dumps(bad))
    assert probes.parse_speed(json.dumps(dict(peak_speed_m_per_s=0.6, reasoning="r")))["peak_speed_m_per_s"] == 0.6
    for bad in (dict(peak_speed_m_per_s=0, reasoning="r"), dict(peak_speed_m_per_s=0.5, reasoning=" "), dict(speed=0.5, reasoning="r")):
        with pytest.raises(ValueError):
            probes.parse_speed(json.dumps(bad))

def test_query_claude_sends_a_probe_set_in_its_own_file(tmp_path):
    calls = []
    def create(**kw):
        calls.append(kw)
        text = json.dumps(dict(peak_speed_m_per_s=0.5, reasoning="energy balance"))
        return NS(content=[NS(type="text", text=text)], stop_reason="end_turn", model="claude-opus-5-5", _request_id="r", usage=NS(input_tokens=1, output_tokens=1))
    client, prompts = NS(messages=NS(create=create)), probes.build("speed", only=["L01"])
    schema, parse = probes.KINDS["speed"]
    path = tmp_path/"speed.json"
    count = Q.fill(path, prompts, query=True, client=client, out=lambda *a: None, schema=schema, parse=parse)
    assert count["answered"] == 3 and calls[0]["output_config"]["format"]["schema"] == probes.SPEED_SCHEMA
    data = json.loads(path.read_text())
    assert set(data["entries"]) == {R.standin_key(d["prompt"]) for d in prompts.values()} and data["settings"]["output_config"]["format"]["schema"] == schema
    with pytest.raises(ValueError, match="another file"):                          # the friction question does not go into a probe's file
        Q.fill(path, R.load_pairs(R.TRUTH, ["L01"]), query=True, client=client, out=lambda *a: None)
    assert Q.request_settings()["output_config"]["format"]["schema"] == llm_prior.SCHEMA      # the default is unchanged

@pytest.mark.slow
def test_prior_study_end_to_end_on_five_pairs():
    ctx = PS.setup({"claude": DATA/"llm_standin_claude_haiku.json"}, R.TRUTH, ["L01", "L05", "L16", "Y01", "Y03"], 800, [0])
    job = {k: v for k, v in ctx.items() if k != "pairs"}
    parts = [PS._pair((p, job)) for p in ctx["truth"]]
    rows, cal = [r for _, rs, _ in parts for r in rs], {p: c for p, _, c in parts}
    sources = {r["source"] for r in rows}
    assert sources == {"generic", "population", "lookup", "robust@lookup", "llm_claude", "generic@claude", "robust@claude", "recal_claude"}
    assert {r["rule"] for r in rows if r["stage"] == "first"} == {"plug-in", "aware", "aware_sq_abs", "edge-aware"}
    assert all(r["fell"] in (0, 1) for r in rows) and len(ctx["targets"]) == 3 and ctx["cap"] > 1.1       # the rerun's setup: the cap with the 6 m/s^2 brake
    first = [r for r in rows if r["stage"] == "first" and r["source"] == "generic" and r["pair"] == "L01"]
    aware, plug = (np.mean([abs(r["miss_mm"]) for r in first if r["rule"] == rule]) for rule in ("aware", "plug-in"))
    assert aware > plug                    # the wide prior aims short: on a typical pair that costs more than trusting the median
    assert cal["L01"]["generic"]["inside"] == 1 and cal["L01"]["generic"]["weak_sd"] < cal["L01"]["generic"]["prior_sd"]      # the weak slide still teaches
    text = PS.summarize(ctx, rows, cal)
    for heading in ("1. THE PRIORS' 90% INTERVALS", "2. FIRST SHOT", "3. TABLE EDGE", "4. AFTER ONE WEAK PRACTICE SLIDE", "5. DIFFERENCES OVER PAIRS"):
        assert heading in text
    assert "llm_claude - generic@claude" in text and "robust@claude - llm_claude" in text
