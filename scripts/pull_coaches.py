#!/usr/bin/env python
"""
pull_coaches.py -- build data/reference/coaches.parquet: one row per
(season, espn_team_id) with the D-I men's basketball head coach.

SOURCE. English Wikipedia, via the MediaWiki Action API
(en.wikipedia.org/w/api.php), fetching the wikitext of each team's per-season
article (e.g. "2024-25 Duke Blue Devils men's basketball team") and parsing
the `{{Infobox college sports team season}}` template's `head_coach` /
`head_coach2` / `head_coach3` fields. Content is CC BY-SA 4.0. See
docs/tests/coaches_table_2026-09-10.md for the full source-compliance
verdict (robots.txt reading, ToS, licensing) and why this beat Wikidata
SPARQL, the CollegeBasketballData API (no coach field), and stats.ncaa.org
for this build.

COMPLIANCE. robots.txt (en.wikipedia.org) disallows generic crawling of
/w/ and /api/ for the default crawler UA, EXCEPT the narrow mobileview
carve-out -- but Wikimedia's own written API policies (API:Etiquette,
Foundation User-Agent policy, WMF API Usage Guidelines) explicitly document
api.php as the sanctioned path for programmatic/bot access, conditioned on:
a descriptive User-Agent with contact info (set below), sequential (not
parallel) requests, and honoring throttling. This script batches up to 50
titles per request and sleeps SLEEP_SECS between requests -- both call
volume and pacing are far inside those bounds. Do NOT point this script at
query.wikidata.org/sparql -- that path IS disallowed for all user agents
with no equivalent carve-out; cross-validation instead uses
www.wikidata.org/w/api.php (wbgetentities / wbgetclaims), which carries the
same api.php sanction as Wikipedia.

SEASON CONVENTION. Ending-year, same as hoopR/CBBD/KenPom: season 2025 is
the 2024-25 season; season 2027 is the upcoming (not yet played) 2026-27
season.

Usage:
    .venv/Scripts/python.exe scripts/pull_coaches.py
    .venv/Scripts/python.exe scripts/pull_coaches.py --seasons 2026 2027

Writes:
    data/reference/coaches.parquet
    data/raw/coaches/unresolved.csv        (season,espn_team_id,team,titles_tried)
    data/raw/coaches/raw_pages.json        (cache of every fetched wikitext, for
                                             debugging/reproducibility)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CROSSWALK_PATH = REPO_ROOT / "data" / "reference" / "team_crosswalk.parquet"
OUT_PATH = REPO_ROOT / "data" / "reference" / "coaches.parquet"
RAW_DIR = REPO_ROOT / "data" / "raw" / "coaches"
UNRESOLVED_PATH = RAW_DIR / "unresolved.csv"
RAW_PAGES_PATH = RAW_DIR / "raw_pages.json"

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = (
    "CBBCleanSheetCoachesBot/1.0 "
    "(https://github.com/cmpeavlerjr72/CBB-redesign; cmpeavlerjr@gmail.com) "
    "python-requests"
)
SLEEP_SECS = 1.0
BATCH_SIZE = 50
SOURCE_NAME = "wikipedia_infobox"

SEASONS_DEFAULT = [2022, 2023, 2024, 2025, 2026, 2027]

# Known Wikipedia season-article title quirks the raw ESPN display name does
# not predict (populated by hand after inspecting search-fallback misses).
# Maps espn_team_id -> the exact "<mascot phrase>" to substitute for the
# crosswalk's espn display name when building "{year} {phrase} men's
# basketball team".
TITLE_NAME_OVERRIDES: dict[int, str] = {
    2511: "Queens Royals",  # ESPN "Queens University Royals"
}

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": USER_AGENT})

_HEAD_COACH_RE = re.compile(r"\n\s*\|\s*head_coach(\d*)\s*=\s*([^\n]*)")
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_INTERIM_RE = re.compile(r"\(\s*interim[^)]*\)", re.IGNORECASE)
_PAREN_RE = re.compile(r"\s*\([^)]*\)")


def season_year_label(season: int) -> str:
    """season 2025 -> '2024-25' (en dash)."""
    start = season - 1
    end2 = season % 100
    return f"{start}–{end2:02d}"


def _clean_title_name(name: str) -> str:
    # Drop a leading "University of "/trailing "University"/"College" that
    # ESPN sometimes includes but Wikipedia season-article titles omit.
    n = name
    n = re.sub(r"\bUniversity\b", "", n)
    n = re.sub(r"\s{2,}", " ", n).strip()
    return n


def candidate_titles(espn_team_id: int, name: str, season: int) -> list[str]:
    label = season_year_label(season)
    names = []
    override = TITLE_NAME_OVERRIDES.get(espn_team_id)
    if override:
        names.append(override)
    names.append(name)
    cleaned = _clean_title_name(name)
    if cleaned != name:
        names.append(cleaned)
    no_paren = _PAREN_RE.sub("", name).strip()
    no_paren = re.sub(r"\s{2,}", " ", no_paren)
    if no_paren and no_paren != name:
        names.append(no_paren)
    seen = set()
    out = []
    for n in names:
        for suffix in ("men's basketball team", "basketball team"):
            t = f"{label} {n} {suffix}"
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


@dataclass
class FetchResult:
    title_requested: str
    title_final: Optional[str] = None
    missing: bool = True
    content: Optional[str] = None


def _api_get(params: dict, retries: int = 3) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    last_exc = None
    for attempt in range(retries):
        try:
            resp = _SESSION.get(WIKI_API, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"MediaWiki API request failed after {retries} tries: {last_exc}")


def fetch_titles_batch(titles: list[str]) -> dict[str, FetchResult]:
    """One action=query call for up to 50 titles. Returns {requested_title: FetchResult}."""
    out: dict[str, FetchResult] = {t: FetchResult(title_requested=t) for t in titles}
    data = _api_get(
        {
            "action": "query",
            "prop": "revisions",
            "rvslots": "main",
            "rvprop": "content",
            "redirects": "1",
            "titles": "|".join(titles),
        }
    )
    query = data.get("query", {})
    # normalization / redirect chains map requested title -> final title
    norm_map = {n["from"]: n["to"] for n in query.get("normalized", [])}
    redir_map = {r["from"]: r["to"] for r in query.get("redirects", [])}

    def resolve(t: str) -> str:
        cur = t
        if cur in norm_map:
            cur = norm_map[cur]
        if cur in redir_map:
            cur = redir_map[cur]
        return cur

    final_to_requested: dict[str, str] = {}
    for t in titles:
        final_to_requested[resolve(t)] = t

    for page in query.get("pages", []):
        title = page.get("title")
        requested = final_to_requested.get(title, title)
        res = out.get(requested)
        if res is None:
            continue
        res.title_final = title
        if page.get("missing"):
            res.missing = True
        else:
            res.missing = False
            revs = page.get("revisions")
            if revs:
                res.content = revs[0]["slots"]["main"]["content"]
    time.sleep(SLEEP_SECS)
    return out


_GENERIC_NAME_TOKENS = {"state", "university", "college", "the", "of", "at", "men", "mens"}


def _significant_tokens(name: str) -> set[str]:
    toks = set(re.findall(r"[a-z0-9]+", name.lower()))
    sig = {t for t in toks if t not in _GENERIC_NAME_TOKENS and len(t) >= 3}
    return sig or toks


def _title_matches_team(title: str, name: str) -> bool:
    """Guard against CirrusSearch relevance-ranking a completely unrelated
    (but well-formed, already-existing) season article to the top of the
    results for a team with no article of its own -- verified bug: searches
    for two dozen mid-major 2026-27 previews returned major programs'
    already-published preview articles (Michigan, Indiana, Purdue, ...)
    that merely matched the generic "{year} ... men's basketball team"
    phrase, with zero relation to the team actually being searched for."""
    title_toks = set(re.findall(r"[a-z0-9]+", title.lower()))
    return bool(_significant_tokens(name) & title_toks)


def search_fallback(season: int, name: str) -> Optional[str]:
    """CirrusSearch fallback: find the season article by full-text search
    when the guessed title doesn't exist."""
    label = season_year_label(season)
    query = f"{label} {name} men's basketball team"
    data = _api_get(
        {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": "5",
        }
    )
    results = data.get("query", {}).get("search", [])
    time.sleep(SLEEP_SECS)
    mens_hit, plain_hit = None, None
    for r in results:
        t = r["title"]
        tl = t.lower()
        if not t.startswith(label):
            continue
        if "women" in tl:
            continue
        if not _title_matches_team(t, name):
            continue
        if tl.endswith("men's basketball team") and mens_hit is None:
            mens_hit = t
        elif tl.endswith("basketball team") and plain_hit is None:
            plain_hit = t
    return mens_hit or plain_hit


def extract_infobox_block(content: str, marker: str = "Infobox college sports team season") -> str:
    m = re.search(r"\{\{\s*" + re.escape(marker), content, re.IGNORECASE)
    if not m:
        return content[:4000]
    start = m.start()
    # The infobox's closing braces are on their own line, but may carry
    # leading whitespace (e.g. "  }}") -- a bare "\n}}" search skips right
    # past that and matches the next unrelated template's close much further
    # down the page (verified bug: swallowed a "Roster/staff" navbox's own
    # bulleted "head_coach=" entry on the 2021-22 Iona Gaels article,
    # producing a spurious phantom `head_coach2`/midseason-change flag).
    end_m = re.search(r"\n[ \t]*\}\}", content[start:])
    end = start + end_m.end() if end_m else min(len(content), start + 4000)
    return content[start:end]


_WOMENS_RE = re.compile(r"American women's college basketball season|women's college basketball team representing", re.IGNORECASE)


def is_womens_page(content: str) -> bool:
    """Guard against a 'basketball team' (no 'men's') title guess/search hit
    resolving to the women's program's article instead."""
    return bool(_WOMENS_RE.search(content[:2000]))


@dataclass
class CoachSlot:
    slot: int
    raw: str
    name: str
    interim: bool
    link_target: Optional[str]


_CURRENT_COACH_RE = re.compile(r"\n\s*\|\s*coach\s*=\s*([^\n]*)")


def parse_current_coach(content: str) -> Optional[CoachSlot]:
    """Parse the `coach` field of the general (non-season) 'Infobox college
    basketball team' template -- used as the season-2027 fallback when a
    team has no dedicated 2026-27 season article yet (this early in the
    season, most low/mid-major programs don't)."""
    block = extract_infobox_block(content, marker="Infobox college basketball team")
    m = _CURRENT_COACH_RE.search(block)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw:
        return None
    interim = bool(_INTERIM_RE.search(raw))
    link_m = _WIKILINK_RE.search(raw)
    link_target = link_m.group(1).strip() if link_m else None
    if link_m:
        full = link_m.group(0)
        pipe = full.find("|")
        disp = full[pipe + 1 : -2] if pipe != -1 else link_target
    else:
        disp = _INTERIM_RE.sub("", raw).strip()
        disp = re.sub(r"'{2,}", "", disp)
    return CoachSlot(slot=1, raw=raw, name=disp, interim=interim, link_target=link_target)


def general_candidate_titles(espn_team_id: int, name: str) -> list[str]:
    names = []
    override = TITLE_NAME_OVERRIDES.get(espn_team_id)
    if override:
        names.append(override)
    names.append(name)
    cleaned = _clean_title_name(name)
    if cleaned != name:
        names.append(cleaned)
    no_paren = _PAREN_RE.sub("", name).strip()
    no_paren = re.sub(r"\s{2,}", " ", no_paren)
    if no_paren and no_paren != name:
        names.append(no_paren)
    seen = set()
    out = []
    for n in names:
        for suffix in ("men's basketball", "basketball"):
            t = f"{n} {suffix}"
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


_YEAR_IN_TITLE_RE = re.compile(r"\d{4}")


def general_search_fallback(name: str) -> Optional[str]:
    query = f"{name} men's basketball"
    data = _api_get({"action": "query", "list": "search", "srsearch": query, "srlimit": "8"})
    results = data.get("query", {}).get("search", [])
    time.sleep(SLEEP_SECS)
    mens_hit, plain_hit = None, None
    for r in results:
        t = r["title"]
        tl = t.lower()
        if "women" in tl or _YEAR_IN_TITLE_RE.search(t):
            continue
        if "statistical leaders" in tl or "roster" in tl or "season" in tl:
            continue
        if not tl.endswith("basketball") and not tl.endswith("men's basketball"):
            continue
        if not _title_matches_team(t, name):
            continue
        if tl.endswith("men's basketball") and mens_hit is None:
            mens_hit = t
        elif tl.endswith("basketball") and plain_hit is None:
            plain_hit = t
    return mens_hit or plain_hit


def parse_head_coaches(content: str) -> list[CoachSlot]:
    block = extract_infobox_block(content)
    slots: list[CoachSlot] = []
    for m in _HEAD_COACH_RE.finditer(block):
        slot_num = int(m.group(1)) if m.group(1) else 1
        raw = m.group(2).strip()
        if not raw:
            continue
        interim = bool(_INTERIM_RE.search(raw))
        link_m = _WIKILINK_RE.search(raw)
        link_target = link_m.group(1).strip() if link_m else None
        # display name: prefer the wikilink's display text, else the link
        # target, else the raw text with the interim tag and any leftover
        # wikitext markup stripped.
        if link_m:
            full = link_m.group(0)
            pipe = full.find("|")
            if pipe != -1:
                disp = full[pipe + 1 : -2]
            else:
                disp = link_target
        else:
            disp = _INTERIM_RE.sub("", raw).strip()
            disp = re.sub(r"'{2,}", "", disp)  # italics/bold markup
        slots.append(CoachSlot(slot=slot_num, raw=raw, name=disp, interim=interim, link_target=link_target))
    slots.sort(key=lambda s: s.slot)
    return slots


def slugify(text: str) -> str:
    s = text.lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "unknown"


def coach_id_for(slot: CoachSlot) -> str:
    key = slot.link_target if slot.link_target else slot.name
    return slugify(key)


def load_team_seasons(seasons: list[int]) -> pd.DataFrame:
    cw = pd.read_parquet(CROSSWALK_PATH)
    rows = []
    for _, r in cw.iterrows():
        names_by_season = json.loads(r["espn_names_by_season"])
        first, last = int(r["first_season"]), int(r["last_season"])
        for season in seasons:
            if season <= 2026:
                if not (first <= season <= last):
                    continue
            else:
                # 2027 (upcoming season): include iff still D-I as of the
                # last season we have schedule data for (2026).
                if last != 2026:
                    continue
            name = names_by_season.get(str(season)) or names_by_season.get(str(last)) or r["espn_name"]
            rows.append(
                {
                    "season": season,
                    "espn_team_id": int(r["espn_team_id"]),
                    "team": r["cbbd_name"],
                    "espn_display_name": name,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=SEASONS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap number of team-seasons")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    team_seasons = load_team_seasons(args.seasons)
    if args.limit:
        team_seasons = team_seasons.head(args.limit)
    print(f"[pull_coaches] {len(team_seasons)} team-seasons to resolve across seasons {args.seasons}", file=sys.stderr)

    # --- pass 1: guessed titles, batched -------------------------------
    guess_map: dict[tuple[int, int], list[str]] = {}
    all_titles: list[str] = []
    for _, r in team_seasons.iterrows():
        key = (int(r["season"]), int(r["espn_team_id"]))
        cands = candidate_titles(int(r["espn_team_id"]), r["espn_display_name"], int(r["season"]))
        guess_map[key] = cands
        all_titles.extend(cands)
    all_titles = list(dict.fromkeys(all_titles))

    fetched: dict[str, FetchResult] = {}
    for i in range(0, len(all_titles), BATCH_SIZE):
        batch = all_titles[i : i + BATCH_SIZE]
        print(f"[pull_coaches] pass1 batch {i // BATCH_SIZE + 1}/{-(-len(all_titles)//BATCH_SIZE)}", file=sys.stderr)
        fetched.update(fetch_titles_batch(batch))

    # Resolve each team-season to a title with real content.
    resolved: dict[tuple[int, int], FetchResult] = {}
    unresolved: list[tuple[int, int, str]] = []
    for key, cands in guess_map.items():
        hit = None
        for t in cands:
            fr = fetched.get(t)
            if fr and not fr.missing and fr.content and "head_coach" in fr.content and not is_womens_page(fr.content):
                hit = fr
                break
        if hit:
            resolved[key] = hit
        else:
            unresolved.append((*key, cands[0]))

    print(f"[pull_coaches] pass1 resolved {len(resolved)}/{len(guess_map)}; searching fallback for {len(unresolved)}", file=sys.stderr)

    # --- pass 2: search fallback for misses -----------------------------
    fallback_titles: dict[tuple[int, int], str] = {}
    still_unresolved: list[tuple[int, int, str]] = []
    ts_lookup = team_seasons.set_index(["season", "espn_team_id"])
    for season, tid, _ in unresolved:
        name = ts_lookup.loc[(season, tid), "espn_display_name"]
        found = search_fallback(season, name)
        if found:
            fallback_titles[(season, tid)] = found
        else:
            still_unresolved.append((season, tid, name))

    if fallback_titles:
        fb_title_list = list(dict.fromkeys(fallback_titles.values()))
        fb_fetched: dict[str, FetchResult] = {}
        for i in range(0, len(fb_title_list), BATCH_SIZE):
            batch = fb_title_list[i : i + BATCH_SIZE]
            fb_fetched.update(fetch_titles_batch(batch))
        for key, title in fallback_titles.items():
            fr = fb_fetched.get(title)
            if fr and not fr.missing and fr.content and "head_coach" in fr.content and not is_womens_page(fr.content):
                resolved[key] = fr
            else:
                season, tid = key
                name = ts_lookup.loc[(season, tid), "espn_display_name"]
                still_unresolved.append((season, tid, name))

    print(f"[pull_coaches] pass2 resolved {len(resolved)}/{len(guess_map)}; unresolved {len(still_unresolved)}", file=sys.stderr)

    # --- pass 3: current-coach fallback for season 2027 (no preseason
    # article yet for most non-marquee programs this early) ---------------
    resolved_current: dict[tuple[int, int], FetchResult] = {}
    remaining: list[tuple[int, int, str]] = []
    season2027_misses = [(s, t, n) for (s, t, n) in still_unresolved if s == 2027]
    other_misses = [(s, t, n) for (s, t, n) in still_unresolved if s != 2027]
    if season2027_misses:
        print(f"[pull_coaches] pass3: current-coach fallback for {len(season2027_misses)} season-2027 misses", file=sys.stderr)
        gen_guess_map: dict[tuple[int, int], list[str]] = {}
        gen_titles: list[str] = []
        for season, tid, name in season2027_misses:
            cands = general_candidate_titles(tid, name)
            gen_guess_map[(season, tid)] = cands
            gen_titles.extend(cands)
        gen_titles = list(dict.fromkeys(gen_titles))
        gen_fetched: dict[str, FetchResult] = {}
        for i in range(0, len(gen_titles), BATCH_SIZE):
            batch = gen_titles[i : i + BATCH_SIZE]
            gen_fetched.update(fetch_titles_batch(batch))

        for season, tid, name in season2027_misses:
            hit = None
            for t in gen_guess_map[(season, tid)]:
                fr = gen_fetched.get(t)
                if fr and not fr.missing and fr.content and _CURRENT_COACH_RE.search(fr.content) and not is_womens_page(fr.content):
                    hit = fr
                    break
            if hit:
                resolved_current[(season, tid)] = hit
            else:
                remaining.append((season, tid, name))

        if remaining:
            gen_fb_titles: dict[tuple[int, int], str] = {}
            still_remaining: list[tuple[int, int, str]] = []
            for season, tid, name in remaining:
                found = general_search_fallback(name)
                if found:
                    gen_fb_titles[(season, tid)] = found
                else:
                    still_remaining.append((season, tid, name))
            if gen_fb_titles:
                fb_title_list = list(dict.fromkeys(gen_fb_titles.values()))
                gen_fb_fetched: dict[str, FetchResult] = {}
                for i in range(0, len(fb_title_list), BATCH_SIZE):
                    batch = fb_title_list[i : i + BATCH_SIZE]
                    gen_fb_fetched.update(fetch_titles_batch(batch))
                for (season, tid), title in gen_fb_titles.items():
                    fr = gen_fb_fetched.get(title)
                    if fr and not fr.missing and fr.content and _CURRENT_COACH_RE.search(fr.content) and not is_womens_page(fr.content):
                        resolved_current[(season, tid)] = fr
                    else:
                        name = next(n for s, t, n in remaining if s == season and t == tid)
                        still_remaining.append((season, tid, name))
            remaining = still_remaining
        print(f"[pull_coaches] pass3 resolved {len(resolved_current)}/{len(season2027_misses)}", file=sys.stderr)

    still_unresolved = other_misses + remaining
    print(f"[pull_coaches] final resolved {len(resolved) + len(resolved_current)}/{len(guess_map)}; unresolved {len(still_unresolved)}", file=sys.stderr)

    # cache raw pages
    raw_cache = {f"{s}|{t}": fr.content for (s, t), fr in resolved.items()}
    raw_cache.update({f"{s}|{t}|current": fr.content for (s, t), fr in resolved_current.items()})
    RAW_PAGES_PATH.write_text(json.dumps(raw_cache, ensure_ascii=False), encoding="utf-8")

    if still_unresolved:
        pd.DataFrame(still_unresolved, columns=["season", "espn_team_id", "team"]).to_csv(UNRESOLVED_PATH, index=False)
    elif UNRESOLVED_PATH.exists():
        UNRESOLVED_PATH.unlink()

    # --- parse + assemble -------------------------------------------------
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for (season, tid), fr in resolved.items():
        team_name = ts_lookup.loc[(season, tid), "team"] if (season, tid) in ts_lookup.index else None
        slots = parse_head_coaches(fr.content)
        if not slots:
            still_unresolved.append((season, tid, team_name))
            continue
        primary = slots[0]
        notes = None
        midseason_change = False
        if len(slots) > 1:
            midseason_change = True
            others = ", ".join(f"{s.name}{' (interim)' if s.interim else ''}" for s in slots[1:])
            notes = f"midseason change: replaced by {others}"
        url_title = fr.title_final.replace(" ", "_")
        rows.append(
            {
                "season": season,
                "espn_team_id": tid,
                "team": team_name,
                "head_coach": primary.name,
                "coach_id": coach_id_for(primary),
                "interim": bool(primary.interim),
                "midseason_change": midseason_change,
                "notes": notes,
                "source": SOURCE_NAME,
                "source_url": f"https://en.wikipedia.org/wiki/{url_title}",
                "fetched_at": fetched_at,
            }
        )

    # season-2027 current-coach fallback rows (general program-page infobox,
    # not a season-specific one -- see pass 3 above).
    for (season, tid), fr in resolved_current.items():
        team_name = ts_lookup.loc[(season, tid), "team"] if (season, tid) in ts_lookup.index else None
        primary = parse_current_coach(fr.content)
        if primary is None:
            still_unresolved.append((season, tid, team_name))
            continue
        url_title = fr.title_final.replace(" ", "_")
        rows.append(
            {
                "season": season,
                "espn_team_id": tid,
                "team": team_name,
                "head_coach": primary.name,
                "coach_id": coach_id_for(primary),
                "interim": bool(primary.interim),
                "midseason_change": False,
                "notes": "current-coach fallback: no 2026-27 season article yet",
                "source": "wikipedia_infobox_current",
                "source_url": f"https://en.wikipedia.org/wiki/{url_title}",
                "fetched_at": fetched_at,
            }
        )

    out = pd.DataFrame(rows).sort_values(["season", "espn_team_id"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False)

    print(f"[pull_coaches] wrote {len(out)} rows -> {OUT_PATH}", file=sys.stderr)
    print(f"[pull_coaches] {len(still_unresolved)} team-seasons unresolved -> {UNRESOLVED_PATH}", file=sys.stderr)
    if still_unresolved:
        pd.DataFrame(still_unresolved, columns=["season", "espn_team_id", "team"]).drop_duplicates().to_csv(
            UNRESOLVED_PATH, index=False
        )
    for season in args.seasons:
        n = int((out["season"] == season).sum())
        expected = int((team_seasons["season"] == season).sum())
        print(f"[pull_coaches]   season {season}: {n}/{expected} filled", file=sys.stderr)


if __name__ == "__main__":
    main()
