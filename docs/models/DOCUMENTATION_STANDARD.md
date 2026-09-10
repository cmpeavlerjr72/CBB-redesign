# Model Documentation Standard

Every model trained for the sim gets its own folder under `docs/models/` with three files:

```
docs/models/{model_name}/
├── model.md         — main model doc (what / why / how / results)
├── features.md      — feature inventory with provenance
└── experiments.md   — full grid results + decisions from data
```

For cascaded models (like the pass outcome chain), nest:

```
docs/models/pass/
├── sack/
├── int/
├── completion/
└── yards/
```

The repo index at `docs/models/README.md` lists all models and links each `model.md`.

---

## Why three files, not one

We want the same audience split as the data pipeline:

- **`model.md`**: the "why does this model exist and how does the sim use it" reader. Stable. Rewritten only when decisions change.
- **`features.md`**: the "exactly what goes in" reader. Includes the source file, computation logic, and fallback values. The sim author needs this to wire inputs correctly.
- **`experiments.md`**: the "show me the data" reader. A log of grids run, what was tried, what worked, what didn't. Append-only.

---

## model.md — required sections

### 1. Purpose
One paragraph. What does this model predict? Where does it feed in the sim flow (cite `docs/MODELING_READINESS.md` §game flow)?

### 2. Target variable
- Exact column / derivation from PBP
- Population filter (which plays/situations are in the training set)
- Class balance / distribution snapshot

### 3. Methodology at a glance
- Data window (seasons)
- Train/test split strategy (explicitly call out "temporal walk-forward — no random split")
- Primary metric and why
- Model families tested

### 4. Winner
- Which model + feature set won
- Headline metrics (log-loss, AUC, calibration, + domain-specific)
- Brief note on what beat the competition

### 5. Robustness check
- Segment-level metrics across relevant dimensions (team tier, conference, matchup type, game situation)
- Flag any segment where calibration or accuracy meaningfully degrades

### 6. Decisions log
Every non-obvious choice and why. Examples:
- "Kept `is_home` and `is_dome` despite low importance because tree models incur no cost."
- "Rejected explicit interaction features because tree grid showed they added no signal."
- "Chose LightGBM over XGBoost despite 0.002 logloss advantage — actually they're tied and LightGBM trains faster."

### 7. Consumption from the sim
Minimal Python snippet showing how the sim loads and calls the model. Include:
- Artifact path
- Feature order (cross-reference `features.md`)
- NaN handling / default values
- Output shape and how to interpret

### 8. Artifacts
Table of every file written by the training script:

| Path | What it is |
|---|---|

### 9. Known gaps / followups
Bullet list. What's explicitly not done, and what would likely help if done.

---

## features.md — required sections

### 1. Source manifest (canonical table)

Every feature that goes into ANY version of the model, with full provenance:

| Feature | Dtype | Source file | Computation | Fallback value | Notes |
|---|---|---|---|---|---|

The source file is the authoritative upstream parquet/CSV. The computation column links to the script that produced it. The fallback is what to use when the source returns NaN (e.g., league mean).

### 2. Feature sets tested

Named bundles (e.g., `A_baseline`, `D_full`, `E_defense`). One table:

| Set name | Included features | Rationale |
|---|---|---|

### 3. Rejected features

Features we considered but explicitly did NOT include, with the reason (e.g., "redundant with EPA ratings per the interactions experiment").

---

## experiments.md — required sections

### 1. Grid configuration

| Dimension | Values |
|---|---|
| Feature sets | A, B, C, ... |
| Models | ... |
| Folds | ... |

### 2. Full results

Full metrics table — every (feature_set, model, fold) row. Keep the CSV alongside for programmatic access.

### 3. Interpretation — what the data said

Per experiment, what did we learn?
- "LightGBM beat logistic by 8% relative logloss — tree structure matters."
- "Feature set E didn't beat D — the 3 new defense features were redundant with ridge EPA."

### 4. Decisions from data

The final call and why. Cross-reference the numbers that justified it.

---

## Style notes

- **Lead with the conclusion**, then the evidence. Read: "We chose LightGBM + D_full with logloss 0.596. See grid for why." NOT: "Here's a grid of 48 numbers, can you find the best one?"
- **Show differences in both absolute and relative terms** when they're small. "0.003 logloss" means nothing until you say "0.003 ≈ 0.5% better calibration → ~0.1% edge on prop bets."
- **Absolute metrics belong in `experiments.md`; relative comparisons belong in `model.md`.** Readers of `model.md` want to know "is this model good"; readers of `experiments.md` want to verify the grid.
- **Don't copy-paste numbers** between files — when they change, use only the primary source in `experiments.md` and summarize in `model.md`.

---

## Update triggers

Rewrite `model.md` / `features.md` when:
- A new model beats the current production model
- A new feature or feature set changes what's loaded
- The sim-consumption code changes
- Data sources upstream change in format

Append to `experiments.md` when:
- Any new grid run happens
- A hypothesis gets tested and rejected (that's valuable data too)

Every model's `model.md` + `features.md` + `experiments.md` should answer in isolation:
- Can someone new rebuild this model from scratch with the same data → same answer? (reproducibility test)
