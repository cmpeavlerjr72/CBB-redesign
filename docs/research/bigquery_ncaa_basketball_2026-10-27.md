# BigQuery `bigquery-public-data.ncaa_basketball` fact sheet (2026-10-27)

Desk lookup via web search and fetch. No account created, no queries run. Facts are cited; anything not confirmed is marked UNVERIFIED. No recommendation is made here.

## 1. Licence and terms

- **Official Google dataset page: NOT RETRIEVED.** The Google Cloud public-datasets page for NCAA basketball returned 404 at `docs.cloud.google.com/bigquery/public-data/ncaa-basketball`. The Marketplace listing could not be parsed. No Google-authored licence clause for this dataset was found. UNVERIFIED.
- Google's general public-dataset page (https://docs.cloud.google.com/bigquery/public-data) says: "Google pays for the storage of these datasets and provides public access to the data using a project." and "You pay only for the queries that you perform on the data." It also says each dataset's details, including terms, are on its Cloud Marketplace page. The dataset-specific terms are therefore not confirmed here.
- Kaggle mirror (https://www.kaggle.com/datasets/ncaa/ncaa-basketball), via search snippet only (page fetch returned title only): the licence is labelled CC0, and a Sportradar copyright notice says the data is "for internal research and testing purposes" and that data "may not be exploited in any way without express approval from Sportradar" (quoted from search summary, not verified verbatim). The same listing says no business or commercial use. The CC0 label conflicts with the Sportradar notice. UNVERIFIED as to exact wording.
- Redistribution, ML and betting clauses: NOT FOUND in any retrieved official source. UNVERIFIED. The Sportradar notice above is the only restriction located, and it is secondary.

## 2. Schema of `mbb_pbp_sr`

- Table name confirmed: `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`. Source: https://minimaxir.com/notebooks/basketball-shots/ and https://minimaxir.com/2018/03/basketball-shots/.
- Shot coordinate columns `event_coord_x` and `event_coord_y` are confirmed as used in queries. Source: https://minimaxir.com/notebooks/basketball-shots/ (casts to integer; `600 - event_coord_y` used to flip y). Units and court frame NOT STATED in that source. The notebook rescales x to 0-94 and y to -50 to 0, which suggests feet on a 94x50 court. This is an inference by the notebook author, not a schema statement. UNVERIFIED.
- Shot type: a `shot_type` field exists (rows filtered `shot_type IS NOT NULL`). Its values are NOT DEFINED in the source. Source: minimaxir notebook. UNVERIFIED values.
- Team basket side: a `team_basket` field with values "right" or "left" is reported. Source: https://medium.com/@amitparikh41/college-basketball-shot-mapping-with-python (secondary). UNVERIFIED.
- Made flag: NO explicit made column was seen in the notebook query. Made status is inferred from `points_scored IS NOT NULL` (COUNTIF). Source: minimaxir notebook. The full schema has NOT been viewed. UNVERIFIED whether a made flag exists.
- Game clock, team and player fields: NOT VERIFIED in any retrieved source. The Sportradar v7 NCAA men's basketball play-by-play field list includes "Event Location X Coordinate" and "Event Location Y Coordinate", plus shot type and made-flag fields. Source: https://developer.sportradar.com/docs/read/basketball/NCAA_Mens_Basketball_v7 (from search summary; field page not fully read). That is the vendor feed, not confirmed as the BigQuery schema.
- Sportradar's change log (https://developer.sportradar.com/basketball/reference/ncaamb-change-log) reports that on 12/1/2025 it updated action-area coordinates. Relevant if older coordinates are compared. Source: search summary only.

## 3. Coverage

- Start season: 2013-14. Sources: https://minimaxir.com/2018/03/basketball-shots/ ("since the 2013-2014 season") and https://minimaxir.com/notebooks/basketball-shots/ ("with the 2013-14 season"). One search summary also says the Google skills lab page describes plays from 2009 and scores from 1996. That claim was not verified on the page itself. A Kaggle search summary says play-by-play and box scores back to 2009. Start-year conflict: 2009 vs 2013 for shot data. UNVERIFIED which applies to `mbb_pbp_sr`.
- End season / freeze: NOT FOUND. The notebook filters `scheduled_date < '2018-03-15'` for its own analysis. That is an analyst filter, not a dataset end date. No authoritative statement on the most recent update was found. UNVERIFIED. The Sportradar change log shows the vendor still active as of 12/2025, but that does not confirm the BigQuery table is updated.
- Games per season: NOT VERIFIED. The minimaxir 2018 post says more than 1 million men's shots; the notebook title says "One Million" shots. A lab-guide search summary says about 4 million game events total, from an older snapshot. UNVERIFIED for current counts.
- Coordinate population: NOT QUANTIFIED. Notebook filters null coordinates, implying some are null. Rate UNVERIFIED.

## 4. Query cost and size

- Free tier: 1 TiB of query processing per month. Source: https://docs.cloud.google.com/bigquery/docs/sandbox (quoted: "You are granted the same free compute limit as the BigQuery free tier of 1 TiB of processed query data each month.").
- Sandbox storage: "a lifetime limit of 10 GiB of storage." Source: same page.
- Sandbox expiry: "all tables, views, and partitions automatically expire after 60 days." Source: same page. Relevant to any copied table.
- Public datasets: the sandbox page says "You can evaluate BigQuery by using the BigQuery sandbox to view and query a public dataset." It does not state a separate public-dataset allowance. The 1 TiB figure is the only limit it gives. Source: same page.
- Size of `mbb_pbp_sr`: NOT VERIFIED. Conflicting secondary sources: one older post reports about 24.8 million rows at 16.2 GB (search summary, UNVERIFIED); a lab guide reports about 4 million events, about 3 GB (search summary, older snapshot, UNVERIFIED). A full scan of a table of several GB fits within 1 TiB per month, but this is not confirmed from the live schema. Check the table Details tab in the console (table size and "last modified") before any extract.

## 5. Billing account or card for the sandbox

- Not required. Quoted from https://docs.cloud.google.com/bigquery/docs/sandbox: "without providing a credit card or creating a billing account for your project." Confirmed from the official sandbox page.
- Sandbox limits apply: streaming, DML and Data Transfer Service are excluded per secondary sources (search summary only). UNVERIFIED on the official page.

## Unresolved items for PM

1. The official dataset licence and terms text (Google Cloud Marketplace or dataset page). Not retrieved.
2. Whether a made flag, game clock, team and player columns exist in the live `mbb_pbp_sr` schema. Needs a console schema view, which needs a sandbox project and is a user decision.
3. Current coverage end and last-updated date.
4. Table size in GB.
