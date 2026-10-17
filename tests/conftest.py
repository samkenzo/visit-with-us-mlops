import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "tourism_project"
for p in (PROJECT, PROJECT / "model_building", PROJECT / "hosting"):
    sys.path.insert(0, str(p))
