import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root, so `import physics` works

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: runs a Drake simulation (seconds); deselect with -m 'not slow'")
