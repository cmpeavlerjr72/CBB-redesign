# Sim Sub-Models — Index

Every sim sub-model has a folder here. See `DOCUMENTATION_STANDARD.md` for the doc layout every folder must follow.

Training order and status (per `docs/FRAMEWORK_PLAN.md` §2.1 cascade order):

| # | Model | Folder | Status |
|---|---|---|---|
| | | | |

---

## How to start a new model

1. Create the folder under `docs/models/`.
2. Copy the three empty templates from `DOCUMENTATION_STANDARD.md`.
3. Build the training script under `scripts/train_{model}.py`.
4. Save artifacts under `data/processed/models/{model}/`.
5. Fill in all three doc files. Don't mark the model "done" until `experiments.md` shows the grid that justifies the decision.
