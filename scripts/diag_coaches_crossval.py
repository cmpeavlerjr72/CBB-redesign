#!/usr/bin/env python
"""
diag_coaches_crossval.py -- cross-validate a random sample of
data/reference/coaches.parquet rows against Wikidata (a second, differently
provenanced source: Wikidata's P286 "head coach" claims on the team item,
with P580/P582 start/end qualifiers, are themselves usually sourced from
stats.ncaa.org coaching-history pages -- see docs/tests/coaches_table_2026-09-10.md).

Uses www.wikidata.org/w/api.php (wbgetentities / wbgetclaims), NOT
query.wikidata.org/sparql -- the latter's robots.txt disallows /sparql for
all user agents with no documented bot carve-out, unlike api.php (see the
compliance note in scripts/pull_coaches.py and the source-verdict doc).

Usage:
    .venv/Scripts/python.exe scripts/diag_coaches_crossval.py --n 40 --seed 20260910
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
COACHES_PATH = REPO_ROOT / "data" / "reference" / "coaches.parquet"
OUT_PATH = REPO_ROOT / "data" / "raw" / "coaches" / "crossval_wikidata.csv"

WD_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = (
    "CBBCleanSheetCoachesBot/1.0 "
    "(https://github.com/cmpeavlerjr72/CBB-redesign; cmpeavlerjr@gmail.com) "
    "python-requests"
)
SLEEP_SECS = 1.0

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})

_YEAR_LABEL_RE = re.compile(r"^\d{4}–\d{2}\s+")
_TRAILING_TEAM_RE = re.compile(r"\s+team$", re.IGNORECASE)


def _api_get(params: dict) -> dict:
    params = {**params, "format": "json"}
    resp = _SESSION.get(WD_API, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(SLEEP_SECS)
    return resp.json()


def general_team_title(source_url: str) -> str:
    """'.../2024-25_Duke_Blue_Devils_men's_basketball_team' -> 'Duke Blue Devils men's basketball'"""
    slug = source_url.rsplit("/", 1)[-1]
    from urllib.parse import unquote

    title = unquote(slug).replace("_", " ")
    title = _YEAR_LABEL_RE.sub("", title)
    title = _TRAILING_TEAM_RE.sub("", title)
    return title.strip()


def wikidata_qid_for_title(title: str) -> Optional[str]:
    data = _api_get({"action": "wbgetentities", "sites": "enwiki", "titles": title, "props": "info"})
    entities = data.get("entities", {})
    for qid, ent in entities.items():
        if qid != "-1" and not ent.get("missing"):
            return qid
    return None


def head_coach_claims(qid: str) -> list[dict]:
    data = _api_get({"action": "wbgetclaims", "entity": qid, "property": "P286"})
    return data.get("claims", {}).get("P286", [])


def entity_label(qid: str) -> Optional[str]:
    data = _api_get({"action": "wbgetentities", "ids": qid, "props": "labels", "languages": "en"})
    ent = data.get("entities", {}).get(qid, {})
    return ent.get("labels", {}).get("en", {}).get("value")


def wikitime_to_year(t: Optional[str]) -> Optional[int]:
    if not t:
        return None
    m = re.match(r"\+(\d+)-", t)
    return int(m.group(1)) if m else None


def best_claim_for_season(claims: list[dict], season: int) -> Optional[dict]:
    """season N covers roughly Aug (N-1) through Jul (N). Pick the claim whose
    [start,end] window overlaps that range; prefer the tightest/most recent."""
    season_start, season_end = season - 1, season
    candidates = []
    for c in claims:
        quals = c.get("qualifiers", {})
        start_t = quals.get("P580", [{}])[0].get("datavalue", {}).get("value", {}).get("time") if quals.get("P580") else None
        end_t = quals.get("P582", [{}])[0].get("datavalue", {}).get("value", {}).get("time") if quals.get("P582") else None
        start_y, end_y = wikitime_to_year(start_t), wikitime_to_year(end_t)
        if start_y is None:
            continue
        if start_y <= season_end and (end_y is None or end_y >= season_start):
            candidates.append((start_y, end_y, c))
    if not candidates:
        return None
    # prefer the most recent start (closest coach assignment to the season)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2]


@dataclass
class CrossvalRow:
    season: int
    espn_team_id: int
    team: str
    our_head_coach: str
    wikidata_title: str
    wikidata_qid: Optional[str]
    wikidata_coach: Optional[str]
    agree: Optional[bool]
    note: str


def name_matches(a: str, b: str) -> bool:
    def norm(s: str) -> str:
        return re.sub(r"[^a-z]", "", s.lower())

    an, bn = norm(a), norm(b)
    if an == bn:
        return True
    # last-name match as a looser fallback
    a_last = re.sub(r"[^a-z]", "", a.lower().split()[-1]) if a.split() else ""
    b_last = re.sub(r"[^a-z]", "", b.lower().split()[-1]) if b.split() else ""
    return bool(a_last) and a_last == b_last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260910)
    args = ap.parse_args()

    df = pd.read_parquet(COACHES_PATH)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    title_cache: dict[str, Optional[str]] = {}
    claims_cache: dict[str, list[dict]] = {}
    label_cache: dict[str, Optional[str]] = {}

    rows: list[CrossvalRow] = []
    for i, r in sample.iterrows():
        title = general_team_title(r["source_url"])
        if title not in title_cache:
            try:
                title_cache[title] = wikidata_qid_for_title(title)
            except Exception as exc:  # noqa: BLE001
                print(f"[crossval] wikidata lookup failed for {title!r}: {exc}", file=sys.stderr)
                title_cache[title] = None
        qid = title_cache[title]

        wd_coach = None
        note = ""
        agree = None
        if qid is None:
            note = "no wikidata item for team"
        else:
            if qid not in claims_cache:
                try:
                    claims_cache[qid] = head_coach_claims(qid)
                except Exception as exc:  # noqa: BLE001
                    print(f"[crossval] claims fetch failed for {qid}: {exc}", file=sys.stderr)
                    claims_cache[qid] = []
            claims = claims_cache[qid]
            claim = best_claim_for_season(claims, int(r["season"]))
            if claim is None:
                note = f"no P286 claim overlapping season {r['season']} ({len(claims)} total claims)"
            else:
                coach_qid = claim["mainsnak"]["datavalue"]["value"]["id"]
                if coach_qid not in label_cache:
                    try:
                        label_cache[coach_qid] = entity_label(coach_qid)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[crossval] label fetch failed for {coach_qid}: {exc}", file=sys.stderr)
                        label_cache[coach_qid] = None
                wd_coach = label_cache[coach_qid]
                if wd_coach:
                    agree = name_matches(r["head_coach"], wd_coach)
                    note = "match" if agree else "MISMATCH"

        rows.append(
            CrossvalRow(
                season=int(r["season"]),
                espn_team_id=int(r["espn_team_id"]),
                team=r["team"],
                our_head_coach=r["head_coach"],
                wikidata_title=title,
                wikidata_qid=qid,
                wikidata_coach=wd_coach,
                agree=agree,
                note=note,
            )
        )
        print(f"[crossval] {i+1}/{len(sample)} {r['season']} {r['team']}: ours={r['head_coach']!r} wikidata={wd_coach!r} -> {note}", file=sys.stderr)

    out = pd.DataFrame([vars(r) for r in rows])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    n_checked = out["agree"].notna().sum()
    n_agree = (out["agree"] == True).sum()  # noqa: E712
    print(f"\n[crossval] checked {n_checked}/{len(out)} rows against Wikidata; agreement {n_agree}/{n_checked}", file=sys.stderr)
    print(f"[crossval] wrote {OUT_PATH}", file=sys.stderr)
    mism = out[out["note"] == "MISMATCH"]
    if not mism.empty:
        print("[crossval] MISMATCHES:", file=sys.stderr)
        print(mism.to_string(), file=sys.stderr)


if __name__ == "__main__":
    main()
