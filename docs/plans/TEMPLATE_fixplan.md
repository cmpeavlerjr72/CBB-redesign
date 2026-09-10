# Fix design — INV-{N}{letter}: {short title}

**Status: DESIGN ONLY.** Nothing in this document has been built, trained, run or
committed. No code, data, model or sim was touched in producing it. Every number
below is either quoted from an existing committed doc (cited) or was measured
by a read-only query over files already on disk (the query is reproduced
inline so it can be re-run).

Author pass: {who, when}. Scope: {one sentence — what this fixplan covers and,
just as importantly, what it deliberately does NOT cover}. If a related but
distinct mechanism exists, file it as a separate INV-{N} so the numbering
stays stable and the two designs don't get conflated.

---

## 1. Problem statement

### 1.1 What was observed

Cite the source (`docs/models/change_ledger.md` row, a gate report, a
dated doc under `docs/tests/`). State the symptom in measured terms —
magnitude, direction, sample size — not narrative.

### 1.2 The named cause

The mechanism believed responsible, stated as a falsifiable claim.

### 1.3 The reframe that changes the fix — MEASURED, and it matters

If the initial diagnosis turned out to be incomplete or wrong, say so here
with the measurement that changed the picture. Skip this subsection if there
was no reframe.

### 1.4 The measurement that sizes the defect

The number(s) that justify spending effort on this fix at all: effect size,
confidence, how many games/players/rows are affected.

### 1.5 Why this produces {the downstream symptom}

Trace the causal chain from the defect to what a user/grader/bettor actually
sees.

---

## 2. What already exists that this fix can stand on

### 2.1 Prior work this can reuse

Any banked, built-but-unshipped, or adjacent artifact that shortens this fix.

### 2.2 Why it was banked / not already shipped, and why that verdict does not settle this investigation

### 2.3 The reusable recipe

Concrete: which script, which function, which table.

---

## 3. Candidate designs

### Candidate A — {name}

What it changes, what it costs, what it risks.

### Candidate B — {name}

### Candidate C — {name} (REJECTED, recorded so it is not re-proposed)

Record rejected candidates with the reason — this is what keeps the same
dead end from being re-walked next cycle.

### Candidate D — {name} (REJECTED as lead, retained as an ablation arm)

---

## 4. Recommendation

Lead with the conclusion: which candidate, at what confidence, with what
residual risk.

---

## 5. Implementation plan

Ordered steps, each naming the script(s) touched or added (respecting the
`scripts/` prefix convention in `scripts/README.md`) and the artifact(s)
produced.

---

## 6. Pre-registered evaluation

Written and committed BEFORE the experiment runs — this is the bake-off rule.

### 6.1 Frames

Which seasons/slates/holdouts, and why (walk-forward, no random split).

### 6.2 Seeds

Seed count and rationale (noise floor this many seeds resolves).

### 6.3 Metrics, at every level (overall / per-game / per-team / per-tercile)

### 6.4 Pass / fail gates

The pre-committed rule that decides SHIP / HOLD / REFUTE. State it in
numbers now, not after the read.

### 6.5 What would falsify the whole hypothesis

---

## 7. Cheap first probes

Fast (minutes, not hours), read-only checks that could kill or strongly
support the hypothesis before the full implementation is built.

### P1 — {probe name} (~{time})

### P2 — {probe name} (~{time})

---

## 8. Deliberate non-goals, known limitations, open questions

---

## 9. One-paragraph summary for the board
