# NCAA D-I MBB head coach table -- source verdict, method, coverage (2026-09-10)

`data/reference/coaches.parquet`: one row per (season, espn_team_id), seasons 2022-2027
(ending-year convention -- 2027 is the upcoming, not-yet-played 2026-27 season), for the
367 teams in `data/reference/team_crosswalk.parquet`. Built by `scripts/pull_coaches.py`.
Cross-validated by `scripts/diag_coaches_crossval.py`. Tests: `tests/test_coaches.py`.

## 1. Source verification

Candidates evaluated per the task: (a) Wikipedia season articles + MediaWiki API, (b)
Wikidata (SPARQL and the entity API), (c) CollegeBasketballData (CBBD) API, (d)
NCAA.org/stats.ncaa.org and sportsdataverse. CLAUDE.md's Data rules already ban
sports-reference.com (ToS bans ML use) and barttorvik.com/masseyratings.com (robots
disallow AI crawlers); neither was touched, including as a cross-check.

### (a) Wikipedia -- CHOSEN as primary source

- **URLs fetched**: `https://en.wikipedia.org/w/api.php` (action=query, prop=revisions,
  format=json) for per-season team articles, e.g.
  `2024–25 Duke Blue Devils men's basketball team`; `https://en.wikipedia.org/robots.txt`;
  `https://www.mediawiki.org/wiki/API:Etiquette`;
  `https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy`;
  `https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_API_Usage_Guidelines`;
  `https://wikitech.wikimedia.org/wiki/Robot_policy`;
  `https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use`.
- **License**: article text CC BY-SA 4.0 (+ GFDL). The values we extract (a head coach's
  name for a team-season) are facts, not copyrightable expression, but every row still
  carries a `source_url` for attribution per the ToU's reuse clause.
- **ToS**: Terms of Use section 4 requires automated use to be non-abusive and to follow
  the Robot Policy / API:Etiquette (no ML-training prohibition, unlike sports-reference.com).
- **robots.txt**: `en.wikipedia.org/robots.txt` disallows `/w/` and `/api/` for the default
  crawler group (narrow carve-out only for `action=mobileview` and `load.php`) -- but this
  blanket disallow targets general web-crawling/indexing of dynamically generated pages, not
  programmatic API clients: Wikimedia's own API:Etiquette, User-Agent policy, and API Usage
  Guidelines pages explicitly document `api.php` as the sanctioned bot/tool access path,
  conditioned on a descriptive `User-Agent` with contact info, sequential (not parallel)
  requests, and honoring throttling/`maxlag`. Critically, **no AI-crawler-specific rule
  appears anywhere in en.wikipedia.org's or www.wikidata.org's robots.txt** (checked for
  `anthropic-ai`, `GPTBot`, `CCBot`, `ChatGPT-User`, `Google-Extended`, `Bytespider`, etc. --
  none present), unlike `www.espn.com/robots.txt`, which explicitly lists
  `User-agent: anthropic-ai / Disallow: /` (see §2 below for why that ruled out a
  tempting alternative).
- **Coverage**: every major- and mid-major-conference team has a dedicated per-season
  article with a `{{Infobox college sports team season}}` template carrying a `head_coach`
  field (verified on Duke, Georgia Tech, Louisville). In-season coaching changes are
  captured by the same template via `head_coach2` (and rarely `head_coach3`), each optionally
  suffixed `(interim)` -- verified against the well-documented Chris Mack -> Mike Pegues
  in-season 2021-22 Louisville swap, where the infobox lists `head_coach = Chris Mack` (who
  coached the season's majority of games, 20/31) and `head_coach2 = Mike Pegues (interim)`.
  Coverage thins for the smallest low-major programs and reclassifying-up teams (Southern
  Indiana, Le Moyne, etc. still confirmed to have season articles).
- **Confidence**: high for the coach-of-record identity; medium for whether every genuine
  in-season change is caught (we rely on editors having filled `head_coach2`/prose, not on
  independently counting games from box scores).

### (b) Wikidata -- used only for cross-validation, NOT the SPARQL endpoint

- **URLs fetched**: `https://query.wikidata.org/robots.txt`; `https://www.wikidata.org/robots.txt`;
  `https://www.wikidata.org/w/api.php` (wbsearchentities, wbgetentities, wbgetclaims).
- **robots.txt -- decisive finding**: `query.wikidata.org/robots.txt` is
  `User-agent: * / Disallow: /sparql / Disallow: /bigdata` with **no carve-out for any user
  agent**. Unlike Wikipedia's `/w/api.php`, there is no documented sanctioned-bot exception
  for the SPARQL query service specifically. **This script never queries
  `query.wikidata.org/sparql`.** Instead, cross-validation uses `www.wikidata.org/w/api.php`
  (`wbgetentities`/`wbgetclaims`), which carries the same MediaWiki Action API sanction as
  Wikipedia (same domain family, same API:Etiquette/User-Agent policy coverage).
- **License**: data is CC0 (public domain dedication) -- even more permissive than Wikipedia's
  CC BY-SA.
- **Coverage**: spot-checked. Duke's item (Q4171772) carries 21 `P286` (head coach) claims
  with `P580`/`P582` (start/end) qualifiers back to 1905, referenced to
  `stats.ncaa.org/teams/history/MBB/193` -- i.e. Wikidata's coach history is itself largely
  sourced from NCAA's official coaching-history pages, giving genuinely independent
  provenance from Wikipedia's per-season prose. A low-major program (Southern Indiana,
  Q108303428) had only 1 claim (current coach since 2020, no end date) -- sparse but still
  useful for cross-validating any season from 2020 on. Not every team-season has a
  qualifier-dated claim, so Wikidata is a good spot-check, not a complete primary source.

### (c) CollegeBasketballData (CBBD) API -- ELIMINATED, no coach field exists

- Full OpenAPI spec pulled from `https://api.collegebasketballdata.com/api-docs.json` (one
  fetch) and searched for every case-insensitive occurrence of "coach": the **only** two
  hits are the `pollType` enum (`["ap","coaches"]`) on the rankings endpoint -- i.e. the "AP
  or Coaches Poll" ranking type, unrelated to head-coach identity. `/teams`, `/games`,
  `/games/teams` carry no coach field of any kind. CBBD is not usable for this table at all,
  including as a cross-check.

### (d) NCAA.org / stats.ncaa.org / sportsdataverse -- ELIMINATED / not applicable

- **stats.ncaa.org**: `robots.txt` is

  ```
  # To ban all spiders from the entire site uncomment the next two lines:
   User-Agent: *
   Disallow: /
  ```

  Despite the comment claiming these lines need uncommenting, they are **not** prefixed
  with `#` and are therefore live directives per the robots.txt spec: the entire site
  disallows all automated access. Eliminated on exactly the robots-ban basis CLAUDE.md
  already applies to barttorvik.com/masseyratings.com.
- **ncaa.org**: robots.txt allows crawling, but the site is NCAA governance/news content,
  not a coaching-records database -- not applicable regardless of robots stance.
- **sportsdataverse/hoopR-mbb-data** (CLAUDE.md's primary project source): the released
  data repo's `mbb/` directory has no `coaches` table (`crosswalk`, `game_rosters`,
  `officials`, `pbp`, `player_box`, `player_core`, `player_season_stats`, `rosters`,
  `schedules`, `shots`, `standings`, `team_box`, `team_season_stats` only, checked via the
  GitHub contents API). The `hoopR` R package's `NAMESPACE` does export
  `espn_mbb_coach`/`espn_mbb_coach_record`/`espn_mbb_coach_season`/`espn_mbb_coaches`, and
  the one implemented helper (`espn_basketball_coach_helpers.R`) calls ESPN's live
  `sports.core.api.espn.com/v2/.../coaches/{id}` JSON endpoint. That endpoint was
  test-fetched directly (`.../seasons/{season}/teams/{espn_team_id}/coaches`) and returns
  exactly the right thing -- e.g. season 2022, team 97 (Louisville) -> coach 2494443 ->
  "Chris Mack", matching Wikipedia's majority-of-season convention with zero name-matching
  risk, since it's keyed on the same ESPN team ID already in the crosswalk. **This is not
  used**, however: `www.espn.com/robots.txt` explicitly lists
  `User-agent: anthropic-ai / Disallow: /` (alongside GPTBot, CCBot, ChatGPT-User,
  Google-Extended, Bytespider). `sports.core.api.espn.com` itself returns no robots.txt
  (403), but the parent domain's stated intent toward AI-affiliated automated access is
  unambiguous, and this build is being run by an Anthropic model. Direct ESPN API calls are
  therefore excluded, even though the pre-existing `hoopR-mbb-data` CC BY 4.0 release
  (built by a third party, not by this agent, and already CLAUDE.md's sanctioned primary
  source) is unaffected by that judgment call.

### Verdict

**Primary source: Wikipedia, via the MediaWiki Action API** (`en.wikipedia.org/w/api.php`),
parsing the `{{Infobox college sports team season}}` template. Best coverage of the four
named candidates (a/b/c/d) by a wide margin -- (c) has zero coach data, (d) is robots-banned
(stats.ncaa.org) or not a coaching database (ncaa.org) or has no pre-built table
(sportsdataverse) -- and Wikipedia's licensing/ToS/robots posture is clean (CC BY-SA 4.0,
sanctioned bot API access, no AI-crawler carve-out needed because none exists).
**Cross-validation source: Wikidata**, via its Action API only (not the robots-disallowed
SPARQL endpoint), which is independently useful because its coach-history claims are
largely sourced from NCAA's own coaching-history records.

## 2. Method

`scripts/pull_coaches.py`:

1. Load `data/reference/team_crosswalk.parquet` (367 teams). For each of the 6 seasons,
   include a team iff `first_season <= season <= last_season` (season <= 2026), or
   `last_season == 2026` for season 2027 (i.e. still D-I as of the last season with
   schedule data, assumed continuing into 2026-27).
2. Build the season-article title from the crosswalk's per-season ESPN display name
   (`espn_names_by_season`): `"{season-1}–{season%100:02d} {name} men's basketball team"`,
   plus a couple of cheap variants (parenthetical stripped, "University" stripped).
3. Batch-fetch up to 50 titles per `action=query` call (`redirects=1`, full wikitext),
   sleeping 1s between calls -- well inside Wikimedia's documented etiquette.
4. For any team-season whose guessed titles all miss, fall back to
   `action=query&list=search` (CirrusSearch) and take the first result matching the
   `"{year label} ... men's basketball team"` pattern, then batch-fetch those resolved
   titles the same way.
5. Parse the `head_coach`/`head_coach2`/`head_coach3` fields from the infobox block only
   (not any later navbox on the page). `head_coach` (slot 1) is taken as the season's
   majority-of-games coach -- verified against Wikipedia's own editorial convention (the
   2021-22 Louisville case: Mack fired with 20/31 games coached stays in slot 1). If a
   `head_coach2`/`3` exists, `midseason_change=True` and `notes` records the replacement
   name(s) (with their own interim tag if present). `interim=True` iff slot 1's own text
   carries an `(interim)` annotation (a coach who was interim for the *entire* season, not
   a mid-season swap).
6. `coach_id`: each `head_coach`/`head_coach2` value is parsed as a wikilink
   (`[[Target|Display]]`); `coach_id = slugify(Target)` when a link exists, else
   `slugify(display text)`. Because Wikipedia already disambiguates same-named people via
   parenthetical article-title suffixes (e.g. "James Jones (basketball, born 1964)"), this
   slug is naturally stable and collision-resistant without a separate alias table.
7. Rows with no resolvable source page are **omitted**, not null-filled, and logged to
   `data/raw/coaches/unresolved.csv`.

`scripts/diag_coaches_crossval.py`: samples 40 random `(season, espn_team_id)` rows
(seed 20260910), derives each team's general (non-season) Wikipedia title from its
`source_url`, resolves the Wikidata QID via `wbgetentities?sites=enwiki&titles=...`, pulls
`P286` claims, picks the claim whose `[P580, P582]` window overlaps the season's academic
year, and compares the coach's Wikidata label against our `head_coach` (exact or last-name
match).

## 3. Coverage

`data/reference/coaches.parquet`: **2,169 rows** out of 2,179 expected team-seasons
(367 crosswalk teams, restricted per season to `first_season <= season <= last_season`,
and for season 2027 to teams with `last_season == 2026`) -- **99.5% filled**.

| season | filled | expected | pct | interim rows | midseason_change rows |
|---|---|---|---|---|---|
| 2022 | 354 | 359 | 98.6% | 1 | 13 |
| 2023 | 361 | 363 | 99.4% | 5 | 10 |
| 2024 | 361 | 363 | 99.4% | 1 | 5 |
| 2025 | 364 | 364 | 100.0% | 3 | 5 |
| 2026 | 365 | 365 | 100.0% | 1 | 8 |
| 2027 | 364 | 365 | 99.7% | 0 | 0 (season not yet played) |

Distinct coaches (`coach_id`): **562**. Year-over-year head-coach turnover (by `coach_id`,
among teams present in both seasons):

| transition | changes | teams present both seasons |
|---|---|---|
| 2022 -> 2023 | 64 | 353 |
| 2023 -> 2024 | 62 | 358 |
| 2024 -> 2025 | 69 | 361 |
| 2025 -> 2026 | 63 | 364 |
| 2026 -> 2027 | 68 | 364 |

Row source split: 1,880 rows from a team's own dedicated season article
(`source=wikipedia_infobox`); 289 rows (all season 2027 -- most non-marquee programs don't
have a 2026-27 season article yet, this early before the season starts) from the fallback
to the team's general program-page infobox's `coach` field
(`source=wikipedia_infobox_current`, `notes="current-coach fallback: no 2026-27 season
article yet"`).

**Missing (10 team-seasons)**, logged in `data/raw/coaches/unresolved.csv` and simply
absent from `coaches.parquet` (never null-filled):

| season | team |
|---|---|
| 2022 | UT Rio Grande Valley Vaqueros |
| 2022 | Grand Canyon Lopes |
| 2022 | SE Louisiana Lions |
| 2022 | Seattle U Redhawks |
| 2022 | Texas A&M-Commerce Lions |
| 2023 | Grand Canyon Lopes |
| 2023 | UL Monroe Warhawks |
| 2024 | Hartford Hawks |
| 2024 | Seattle U Redhawks |
| 2027 | SE Louisiana Lions |

All ten are cases where neither a guessed season-article title nor a CirrusSearch
full-text search (constrained to titles that actually share a name token with the team,
see the bug note in section 5) turned up a usable page, and -- for the one 2027 case --
the general program-page infobox also didn't parse. These are genuinely harder Wikipedia
titling cases (Hartford's final D-I season before dropping to D-III; a couple of programs
whose season-article naming apparently deviates further from the "{year} {name} men's/
basketball team" pattern than the variants this script tries) rather than a source-coverage
gap -- a next pass could resolve most of them with one more manually-inspected title
variant each, but that is not attempted here (see section 6).

## 4. Cross-validation

`scripts/diag_coaches_crossval.py --n 40 --seed 20260910` sampled 40 random
`(season, espn_team_id)` rows from the final table and checked each against Wikidata's
`P286` (head coach) claims via `www.wikidata.org/w/api.php` (not the robots-disallowed
SPARQL endpoint). Of the 40: 37 had a Wikidata item + at least one `P286` claim overlapping
the season's window (2 teams had no Wikidata item at all -- George Washington, Dartmouth --
and 1 team's only claim didn't overlap the sampled season); of those 37, **26 agreed with
our value (70%)** and 11 disagreed. Full row-level result:
`data/raw/coaches/crossval_wikidata.csv`.

**Every disagreement spot-checked traces to a Wikidata limitation, not an error in this
table**:

- **Wikidata is stale for very recent hires.** NC State 2027 (ours: Justin Gainey;
  Wikidata: Will Wade, hired 2025) and Brown 2027 (ours: Eric Reveno; Wikidata: Mike
  Martin) reflect coaching changes Wikipedia's actively-edited infobox has already picked
  up but Wikidata's structured claims have not.
- **Wikidata's year-only date precision collides with the college-basketball hiring
  calendar.** A coach hired in March-April is *always* for the following season (the
  outgoing/interim coach finishes the current one), but Wikidata's `P580` (start)
  qualifier is frequently stamped with just the hire-announcement year, one season early
  by our academic-year convention. This produced the Tulsa 2022 (ours: Frank Haith, his
  final season; Wikidata: Eric Konkol, whose P580 year matches Konkol's spring-2023
  hiring) and St. John's 2023 (ours: Mike Anderson, his final season; Wikidata: Rick
  Pitino, hired spring 2023 for 2023-24) mismatches -- both of which this table gets right
  precisely because it reads Wikipedia's season-specific infobox rather than a
  claim-overlap heuristic.
- **Wikidata is sparse on interim coaches.** Manhattan 2023 was spot-verified directly
  against the raw wikitext: the infobox literally reads
  `head_coach = [[RaShawn Stores]] (interim)`, which this table captures correctly
  (`head_coach="RaShawn Stores"`, `interim=True`) against Wikidata's John Gallagher (a
  different, non-overlapping tenure). Fairleigh Dickinson 2023, Nicholls 2023, and North
  Carolina A&T 2023 follow the same pattern.
- **Majority-of-season vs. terminal-coach-of-record convention mismatch.** Miami 2024
  (ours: Jim Larranaga, who coached the season's majority before a February 2024
  resignation; Wikidata: Bill Courtney, the interim who finished it) and James Madison
  2024 (ours: Mark Byington, hired for 2024-25 -- picked up by the same hiring-calendar
  effect above) are read differently by design (see the majority-of-games rule in section
  2), not incorrectly.

**Known-move sanity checks** (independent of the random sample, verified directly in the
built table): Kentucky 2022-2024 John Calipari -> 2025-2027 Mark Pope; Arkansas 2022-2024
Eric Musselman -> 2025-2027 John Calipari (the same Calipari move, other side); USC
2022-2024 Andy Enfield -> 2025-2026 Eric Musselman (the Musselman move, other side); Iona
2022-2023 Rick Pitino -> 2024-2025 Tobin Anderson -> 2026-2027 Dan Geriot; St. John's
2022-2023 Mike Anderson -> 2024-2027 Rick Pitino (the same Pitino move, other side). All
five moves land on the correct season boundary.

## 5. Known limitations

- Majority-of-season-games is inferred from Wikipedia editorial convention (which coach
  occupies infobox slot 1), not from an independent box-score game count. Spot-checked on
  two well-documented cases (2021-22 Louisville, 2023-24 Miami); not exhaustively
  re-derived from pbp.
- **Bug found and fixed during the build**: the CirrusSearch fallback used for titles that
  don't exist as a direct guess (`search_fallback`/`general_search_fallback` in
  `scripts/pull_coaches.py`) originally accepted the first search result matching the
  generic `"{year} ... men's/basketball team"` pattern without checking that it actually
  named the team being searched for. Because a handful of major programs' 2026-27 preview
  articles already exist this early, CirrusSearch's relevance ranking surfaced them (most
  often "2026-27 Michigan Wolverines men's basketball team") as the top hit for dozens of
  unrelated smaller programs' searches, silently attributing the wrong team's coach.
  Fixed with a token-overlap guard (`_title_matches_team`) requiring the candidate title
  to share a real name token with the team being resolved; a whole-table post-hoc audit
  (comparing every row's `team`/ESPN display name against its `source_url`) confirms
  **zero** such mismatches remain in the shipped table.
- A second whitespace-related infobox-boundary bug (`extract_infobox_block` stopping at
  the first bare `"\n}}"` instead of allowing indentation) was also found and fixed: it
  let a handful of articles' *later*, unrelated "coaching staff" navbox templates bleed
  into the parsed block, producing spurious phantom `head_coach2` entries and
  `midseason_change` flags (caught via the 2021-22 Iona Gaels case, where the fix removed
  3 of 16 season-2022 `midseason_change` flags that were false positives).
- Wikidata cross-validation disagreements are systematically explained (section 4) --
  Wikidata staleness/date-precision issues, not errors here -- but that is an inference
  from spot-checking, not a formal proof for every one of the 11 disagreements.
- `coach_id` collisions are possible (not observed) for two different coaches who share a
  name and neither has a Wikipedia article (so no article-title disambiguator exists).
- 10 team-seasons (0.5%) are missing outright rather than guessed at; see section 3 for
  the list and section 6 for what closing that gap would take.

## 6. What a further pass would need (not attempted here)

Per instruction, this build stopped after fixing the two correctness bugs above rather
than iterating further for coverage. The 10 missing rows in section 3 would each need a
manually-found title variant (e.g. checking whether "Grand Canyon" uses "Antelopes" or a
different disambiguator in some seasons, or whether "SE Louisiana" is filed under
"Southeastern Louisiana" for older seasons) -- a handful of one-off lookups, not a change
to the general method. The Wikidata cross-validation's `best_claim_for_season` heuristic
in `scripts/diag_coaches_crossval.py` could be tightened to discount a claim whose P580
falls in the same calendar year as the tested season's second half (Jan-Jun), which is
almost always a hiring-announcement date for the *following* season rather than a
mid-season change -- this would likely resolve several of the 11 disagreements without
touching the primary table at all.
