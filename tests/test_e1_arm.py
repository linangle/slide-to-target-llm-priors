"""e1_arm_validation.py: the job list, one row end to end, and the summary on that row."""
import math
from e1_arm_validation import CASES_A, FIELDS, jobs, run, summarize

def test_job_list():
    js = jobs()
    assert sum(j["part"] == "A" for j in js) == 2*len(CASES_A) == 24
    assert {j["pusher"] for j in js if j["part"] == "A"} == {"arm", "plate"}
    assert all(0.0005 <= j["gap"] <= 0.003 and abs(j["lateral"]) <= 0.003 for j in js if j["part"] == "C")
    assert len(jobs(quick=True)) == 5

def test_one_target_row_and_summary():
    row = run(dict(part="B", case="B_test", pusher="arm", shape="box", mu=0.3, J=100.0, x_target=0.64))
    assert set(FIELDS) <= set(row)
    assert abs(row["miss_mm"]) < 2.0 and row["separated"] == 1 and row["point_contacts"] == 0
    assert row["v_lim_frac"] < 1.0                     # 0.64 m at mu 0.3 is within the iiwa's joint limits
    assert "within joint limits: 1 slides" in summarize([row])
