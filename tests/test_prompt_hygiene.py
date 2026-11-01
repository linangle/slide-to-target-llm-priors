"""Test 7: the prompt builder reads only description_* and photo_* columns; any other column reaching a prompt fails."""
import csv
import pytest
import llm_prior as L

def test_only_description_and_photo_columns_reach_the_prompt():
    with open("ground_truth_placeholder.csv", newline="") as f:
        header = next(csv.reader(f))
    row = {k: f"SENTINEL_{k}" for k in header}        # a unique marker in every column
    row.update(description_object="a PLA block", description_surface="melamine shelf", photo_object="obj.jpg", photo_surface="")
    text = L.build_prompt(row)
    for k in header:
        if not k.startswith(L.PROMPT_PREFIXES):
            assert f"SENTINEL_{k}" not in text, k
    assert "a PLA block" in text and "melamine shelf" in text and "obj.jpg" in text

def test_builder_refuses_a_leak():
    bad = L.PROMPT_TEMPLATE
    L.PROMPT_TEMPLATE = L.PROMPT_TEMPLATE + "mu is about {description_object}"   # template still only names allowed fields...
    try:
        row = {"description_object": "0.42", "mu_k_nominal": "0.42", "description_surface": "x"}
        with pytest.raises(ValueError):                                          # ...but the value of mu_k_nominal appears in the text
            L.build_prompt(row)
    finally:
        L.PROMPT_TEMPLATE = bad

def test_fields_are_whitelisted():
    assert set(L.prompt_fields({"mu_k_nominal": 1, "description_object": "a", "photo_surface": "b", "notes": "c"})) == {"description_object", "photo_surface"}
