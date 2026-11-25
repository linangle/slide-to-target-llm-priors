"""Offline tests of the pair-set runner and the handbook truth file: no Drake run, no API."""
import csv, json, math, sys
from pathlib import Path
import pytest
for _d in ("src", "evaluation", "friction"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/_d))
import make_handbook_truth as H
import run_pair_set as R
from llm_prior import build_prompt, prompt_key, request_settings
from paths import DATA


def test_handbook_truth_file_is_current_and_in_range():
    """The saved file is what the generator writes, has the interface's 21 columns, and stays in the simulated range."""
    with open(DATA/"ground_truth_handbook.csv", newline="") as f:
        reader = csv.DictReader(f)
        saved = list(reader)
        assert reader.fieldnames == H.FIELDS and len(H.FIELDS) == 21
    assert saved == [{k: str(v) for k, v in r.items()} for r in H.rows()]
    assert [r["pair_id"] for r in saved] == [f"L{i:02d}" for i in range(1, 21)]
    with open(DATA/"friction_lookup.csv", newline="") as f:
        lookup = {r["pair_id"]: r for r in csv.DictReader(f)}
    for r in saved:
        assert r["mu_k_nominal"] == lookup[r["pair_id"]]["mu_k_nominal"] and r["in_sim_range"] == "1"
        assert 0.10 <= float(r["mu_k_nominal"]) <= 0.50


def test_handbook_prompts_carry_only_the_descriptions():
    """build_prompt on a whole row raises if anything but a description reaches the prompt; and no number is in one."""
    for r in H.rows():
        text = build_prompt(r)
        assert r["description_object"] in text and r["description_surface"] in text
        assert not any(ch.isdigit() for ch in (r["description_object"] + r["description_surface"]).replace("6/6", ""))
        assert " a a" not in text and " a e" not in text and " a u" not in text.replace("a UHMW", "")


def test_pairs_load_with_yu_descriptions_and_duplicates_are_refused(tmp_path):
    pairs = R.load_pairs()
    assert len(pairs) == 23 and pairs["Y03"]["mu"] == 0.26 and pairs["L08"]["mu"] == 0.14
    assert "A plywood sheet" in pairs["Y03"]["prompt"] and "stainless-steel" in pairs["Y03"]["prompt"]
    assert list(R.load_pairs(only=["L02", "Y01"])) == ["L02", "Y01"]
    with pytest.raises(KeyError):
        R.load_pairs(only=["L99"])
    with pytest.raises(ValueError):
        R.load_pairs([DATA/"ground_truth_handbook.csv", DATA/"ground_truth_handbook.csv"])
    bare = tmp_path/"bare.csv"
    bare.write_text("pair_id,object_material,surface_material,mu_k_nominal,description_object,description_surface\nQ01,a,b,0.3,,\n")
    with pytest.raises(ValueError):
        R.load_pairs([bare])


def test_both_cache_formats_are_read(tmp_path):
    pairs = R.load_pairs()
    standin = DATA/"llm_standin_claude_haiku.json"
    for pid, d in pairs.items():
        est = R.llm_answer(d["prompt"], standin)
        assert est is not None and 0 < est["low"] < est["median"] < est["high"], pid
    assert R.llm_answer("a prompt nobody asked", standin) is None
    assert "Haiku" in R.cache_model(standin)
    gpt = DATA/"llm_cache_published_pilot.json"
    assert R.cache_model(gpt) == "gpt-5.4-mini-2026-03-17"
    assert R.llm_answer(pairs["Y01"]["prompt"], gpt)["median"] == 0.28 and R.llm_answer(pairs["L01"]["prompt"], gpt) is None
    # a project-format cache from another model is read by prompt; two models in one file are refused
    prompt, raw = pairs["L01"]["prompt"], json.dumps(dict(median=.4, low=.3, high=.5, reasoning="r"))
    other = dict(request_settings(), model="some-other-model")
    one = tmp_path/"other.json"
    one.write_text(json.dumps({"k1": dict(prompt=prompt, settings=other, response=raw, cache_version=2)}))
    assert R.cache_model(one) == "some-other-model" and R.llm_answer(prompt, one)["median"] == .4
    two = tmp_path/"two.json"
    two.write_text(json.dumps({"k1": dict(prompt=prompt, settings=other, response=raw, cache_version=2),
                               prompt_key(prompt): dict(prompt=prompt, settings=request_settings(), response=raw, cache_version=2)}))
    with pytest.raises(ValueError):
        R.llm_answer(prompt, two)
    with pytest.raises(ValueError):
        R.cache_model(two)


def test_summary_from_saved_rows(tmp_path):
    """The summary reads a run folder: per-pair first shots, the coverage mark, and the after-practice table."""
    info = {"L01": dict(mu=.42, object="hard steel block", surface="mild steel plate", truth_file="ground_truth_handbook.csv", prompt="p"),
            "Y01": dict(mu=.14, object="stainless steel block", surface="ABS sheet", truth_file="ground_truth_published.csv", prompt="q")}
    settings = dict(pairs=["L01", "Y01"], pair_info=info, arm=dict(layout="side"), peak_cap=1.019, practice_peak=1.01, targets=[.6, .7], seeds=[0],
                    x0=.55, slides=[0, 1], generic_median=.2966, truth_files=["ground_truth_handbook.csv", "ground_truth_published.csv"],
                    llm={"claude": dict(model="m", cache="c.json")},
                    llm_answers={"L01:claude": dict(median=.45, low=.3, high=.65), "Y01:claude": dict(median=.35, low=.2, high=.5)},
                    lookup={"L01": dict(mu_tab=.42, s_tab_log=.246), "Y01": None})
    (tmp_path/"settings.json").write_text(json.dumps(settings))
    err = {"generic": 40., "lookup": 3., "llm_claude": 8., "oracle": .2}
    with (tmp_path/"results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=R.FIELDS); w.writeheader()
        for pair in info:
            for n, seed in ((0, -1), (1, 0)):
                for target in (.6, .7):
                    for m, e in err.items():
                        e = e if n == 0 else .5
                        w.writerow(dict(pair=pair, seed=seed, target_x=target, method=m, slides=n, mu_true=info[pair]["mu"], mu_median=.3, mu_low=.2,
                                        mu_high=.4, peak_speed=.5, at_cap=False, x_rest=target + e/1e3, y_rest=0., error_x_mm=e, valid=True, separated_or_limits=""))
    text = R.summarize(tmp_path)
    assert "0.35 [0.20, 0.50] x" in text and "0.45 [0.30, 0.65]  " in text          # Y01's truth (0.14) is outside the interval
    assert "llm_claude against generic, all pairs: closer on 2 of 2 pairs; mean change -32.0 mm" in text
    assert "handbook pairs" in text and "32 scored shots; 0 at the speed cap" in text
    line = [l for l in text.splitlines() if l.startswith("     1 ")][0].split()
    assert line[1:5] == ["0.50"]*4 and line[5] == "0.00"
    assert (tmp_path/"summary.txt").read_text() == text + "\n"


def test_llm_comparison_rows_and_between_two():
    """compare_llms: an LLM is listed for the pairs it answered, the between-two line uses only the pairs both answered,
    and a prior centred on the truth makes a smaller first shot than one that is not."""
    import compare_llms as C
    c = C.collect({"claude": DATA/"llm_standin_claude_haiku.json", "gpt": DATA/"llm_cache_published_pilot.json"}, only=["L06", "L15", "Y01", "Y02"])
    assert set(c["rows"]["claude"]) == {"L06", "L15", "Y01", "Y02"} and set(c["rows"]["gpt"]) == {"Y01", "Y02"}
    assert c["rows"]["claude"]["L06"]["median"] == 0.25 == c["pairs"]["L06"]["mu"]
    assert c["rows"]["claude"]["L06"]["miss"] < 10 < c["rows"]["claude"]["L15"]["miss"]        # L15: 0.40 stated, 0.15 true
    assert c["rows"]["lookup"]["L06"]["miss"] < c["rows"]["generic"]["L06"]["miss"]
    text = C.report(c)
    assert "claude and gpt, 2 pairs both answered" in text and "0.40 [0.25, 0.60] x" in text
