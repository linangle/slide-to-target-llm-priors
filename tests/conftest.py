import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for d in ("src", "experiments", "evaluation", "friction", "figures"):     # so `import physics`, `import e1_arm_validation`, ... work
    if os.path.join(ROOT, d) not in sys.path:                           # without the editable install too (pyproject.toml does the same)
        sys.path.insert(0, os.path.join(ROOT, d))

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a Drake simulation (seconds); deselect with -m 'not slow'")
