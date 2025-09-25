# Survivor Pool repo reorganized

Layout:
- src/: Python source files (scrape_sked.py, solver.py, solver_historical.py, winprob.py)
- data/: CSV datasets (nfl_schedule_2024.csv, ratings_2024.csv, ...)
- scripts/: small runner scripts such as `run_solver.py`

How to run:
1. Ensure dependencies are installed (pandas, pulp, matplotlib, beautifulsoup4, requests).
2. From the repo root, run:

    python scripts/run_solver.py

Notes:
Canonical copies:
- `data/` now contains canonical CSV datasets (copied from repo root):
    - `nfl_schedule_2024.csv`, `nfl_schedule_2025.csv`, `ratings_2024.csv`, `ratings_2025.csv`, `team_week_win_prob.csv`, `team_week_win_prob_2024.csv`, `cheatsheet.csv`, `survivor_picks_2024.csv`.
- `docs/originals/Strategy Doc.xlsx` contains a copy of the Strategy Doc (original left in repo root as a backup).

Notes / next steps:
- Originals were preserved at the repo root for safety. If you confirm, I can remove the root duplicates and keep the canonical copies only.
- I can update the `src/` scripts to reference `data/` paths (recommended) so the code consistently reads from `data/` instead of the repo root.
