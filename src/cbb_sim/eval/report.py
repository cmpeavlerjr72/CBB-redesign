"""report.py -- shared markdown rendering for the eval/market scripts.

Generalises `scripts/grade_control.py`'s `md_table` helper (the prototype).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_INT_LIKE_COLS = (
    "n", "n_games", "n_team_games", "n_with_line", "n_ml", "n_clv", "n_boot", "n_games_",
    "decile", "quintile", "month", "wins", "losses", "pushes", "seeds", "games",
    "season", "test_season", "seed_offset", "n_seeds",
)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> list[str]:
    if df is None or len(df) == 0:
        return ["(no rows)"]
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (bool, np.bool_, int, np.integer)):
                cells.append(str(v))
            elif isinstance(v, (float, np.floating)):
                if not np.isfinite(v):
                    cells.append("n/a")
                elif float(v).is_integer() and c in _INT_LIKE_COLS:
                    cells.append(str(int(v)))
                else:
                    cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def gate_section(gate_result, level: int = 2) -> list[str]:
    """Render one `gates.GateResult` as `## G<n> -- <title>` ending in the
    gate's literal overall status, with a checks table and any detail
    tables."""
    hh = "#" * level
    L = [f"{hh} {gate_result.gate} -- {gate_result.title}", ""]
    checks_df = pd.DataFrame([
        {"quantity": c.quantity, "value": c.value, "target": c.target,
         "tolerance": c.tolerance, "status": c.status}
        for c in gate_result.checks
    ])
    L += md_table(checks_df) + [""]
    for note in gate_result.notes:
        L += [note, ""]
    for heading, tab in gate_result.tables:
        L += [f"{hh}# {heading}", ""]
        L += md_table(tab) + [""]
    L += [f"**{gate_result.gate} overall: {gate_result.status}**", ""]
    return L
