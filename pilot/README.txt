LOCAL FEASIBILITY PILOT — no cloud calls

Tested with Python 3.14.3.
Install requirements.txt in an isolated environment if needed.
From the project root run:
  python pilot/feasibility.py
  python pilot/hardware_readiness.py
  python pilot/controls.py
  python pilot/package_findings.py

Read research/research_brief.html for assumptions, limits, baselines and the abstract.
All data are synthetic toy instances. No hardware results are included.
The .joblib artifact is generated locally; regenerate it from source rather than loading untrusted model files.
