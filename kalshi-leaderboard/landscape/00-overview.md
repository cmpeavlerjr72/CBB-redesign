# 00 - Kalshi leaderboards: can we see, track and copy the top traders?

Research date 2026-11-02. Everything marked VERIFIED was fetched from public endpoints with no
login; raw pulls are in `../samples/`. Reports 01 and 02 hold the detail and the evidence.

## 1. Short answers

| Question | Answer |
|---|---|
| Can we see who the top traders are? | **Yes.** `GET api.elections.kalshi.com/v1/social/leaderboard` returns up to 60 handles per board with a value, for metrics projected_pnl, volume and num_markets_traded over daily, weekly, monthly, yearly and all_time windows, with an optional category filter. No login. Updates about every five minutes. VERIFIED. |
| Can we see what they trade? | **Only if they opted in.** Kalshi's social layer has three sharing levels. "Full" users expose their fills through `GET /v1/social/trades?nickname=`. "Limited" users show profit and volume but no trades. "Minimum" users show nothing. In a sweep of 141 top handles (top 60 by monthly profit, all-time profit and all-time volume): 5 Full, 112 Limited, 24 Minimum. All five visible ones trade sports or 15-minute Bitcoin markets. Everyone visible on politics, economics, weather or culture is hidden. VERIFIED. |
| Can we track them over time? | **Partly, and only if we build the collector ourselves.** There is no leaderboard history endpoint. But `/v1/social/profile/metrics?nickname=` returns profit and volume for any handle, hidden or not, so daily snapshots build a profit time series per handle. For Full users, `social/trades` pages by cursor and the global feed tags about 9 percent of all prints with a maker or taker nickname. VERIFIED. |
| Can we see positions? | **No.** Positions need a logged-in viewer and the target's permission. Not tested. |
| Does the official trade API or WebSocket carry identity? | **No.** `/trade-api/v2/markets/trades` and the public `trade` channel have no user field. The RFQ `creator_id` hash that kalshi-rfq already profiles is a separate pseudonymous id with no public link to a nickname. VERIFIED. |
| Can we reverse-engineer their strategies? | **Strategy class, sometimes. Exact trades, rarely.** See section 3. |
| Can we copy them? | **Not by mirroring.** Fills arrive after the fact, intent and resting orders are invisible, and 96 percent of top earners hide fills. Imitating a strategy class once identified is realistic. |

## 2. What today's boards look like (VERIFIED 2026-11-02)
- All-time profit: top handle about $7.7M, second about $7.3M, fifth about $4.6M. Top 60 are all seven figures or high six.
- Monthly profit: leader about $342K, tenth about $121K.
- Weekly profit: leader about $62K and trade-hidden. Daily and weekly boards returned identical rows, so one is likely an alias or bug.
- Volume board: the 24 Minimum-sharing handles all sit here, consistent with market makers and institutions (Susquehanna was announced as Kalshi's first committed market maker in April 2024; Jump Trading reportedly took a stake in February 2026) who show volume but not profit.
- The projected_pnl field's semantics are unverified: one handle ranks 17th monthly with a negative all-time figure, so it may be unrealized-inclusive or window-scoped in a way the docs do not state.

## 3. How far reverse-engineering can go

### For the five percent who share fills
Each fill carries ticker, price, size, taker side and action, and timestamp. From a few hundred fills per handle you can classify maker versus taker share, fill-size signature, market series mix, time-of-day and event-relative timing, and whether entries cluster right after public information (scores, data releases). Report 02 did this for the visible handles: one is an apparent 15-minute Bitcoin bot (median fill 100 contracts, 62 percent taker), one is a college-football totals and spreads specialist with about $963K all-time, one is a taker spread across esports and college basketball. Positions can be reconstructed by summing fills.

### For the 80 percent who show profit but hide fills
The lever is the profit time series. Snapshot `profile/metrics` for every top handle several times a day and difference it. Profit jumps that land on CPI mornings, FOMC afternoons, jobs Fridays, NFL Sundays, election nights, or Bitcoin 15-minute cadences identify the category and the event sensitivity without a single visible trade. The leaderboard's category filter sharpens this: a handle in the overall top 60 that is absent from the Sports top 60 with the same profit is a non-sports earner. Neither of these was run today; both are straightforward with the endpoints already mapped.

### From the anonymous tape
Report 02's experiment on four settled NFL and MLB game markets: the top one percent of prints carried 40 to 59 percent of volume, prints of 1,000 or more contracts carried 57 to 76 percent, and the median print was 16 to 37 contracts. One exact size (935.19 contracts) printed 203 times in a single market, always a taker buying YES at one cent, with a median gap of 56 ms, and reappeared in a second market at the same session end. That is an algorithm's signature, visible without any identity field. Combined with the `orderbook_delta` WebSocket channel (not captured today), size-and-timing clustering can separate the two or three large actors in any liquid market and measure whether their flow is maker or taker and whether it wins on markout. It cannot name them.

### What the press knows
Almost nothing specific. No profile of a named top Kalshi trader was readable; a Bloomberg feature from 2026-10-13 on Kalshi's top inflation trader exists but was not retrievable from this machine (worth reading in a browser). A Federal Reserve Board note found Kalshi CPI market forecasts beat the Bloomberg consensus; one vendor backtest found no weather-market edge after fees. Third-party trackers (kalshispy, rivo, Oddpool, Bot for Kalshi, an Apify leaderboard scraper) exist, their numbers disagree with Kalshi's own board, and none use the nickname trade feed, which looks under-exploited.

## 4. What "copying" realistically means
1. **Strategy-class imitation.** Identify the category and mechanics (maker versus taker, event timing, size profile) of a cluster of top handles, then build that strategy with your own model. This is feasible for the sharing handles and, via profit-calendar fingerprinting, for the hidden ones.
2. **Trade mirroring.** Only for Full users, only after the fact, and on markets where a lagged copy still has edge (slow-moving politics or long-dated markets, not 15-minute Bitcoin or in-play sports). Low value.
3. **Counter-flow.** The tape experiment suggests the more tractable use of identity-free data is the opposite of copying: knowing when a large taker algorithm is sweeping at one cent tells a maker where not to rest.

## 5. Terms and etiquette
- The Member Agreement PDF has no scraping, bot or copy-trading clause. The website Terms of Use and Developer Agreement are client-rendered and were not read (HTTP 429); read them before running a collector.
- Sharing is opt-in per user, so Full users' feeds are intended to be public. Hidden users' profit figures are also served without login.
- About 250 requests at two per second drew no blocks.

## 6. Collector (built 2026-11-02, see `../collector/` and the README)
Probes before building settled four design questions: the leaderboard category filter works for
11 categories (Politics, Sports, Crypto, Economics, Mentions, Financials, Elections, Companies,
World, Entertainment, Health), so hidden traders get a category directly from which boards they
appear on; `profile/metrics` returns profit for hidden users too (units are ten-thousandths of a
dollar); a Full user's fill feed paged back six months; and the global trade tape is keyless with
a cursor. The collector snapshots every board every 10 minutes, builds a profit time series per
handle (15-minute cadence for the top 20 of any profit board, 2-hour for the rest), backfills and
follows Full-sharing fills, keeps the named slice of the social feed, archives the raw global tape
with 5-minute per-market aggregates, and snapshots series and markets for the category map and
event calendar. First snapshot: 180 boards, 2,595 distinct handles.

What to look at after a week: `report.py movers` for who made money in which window,
`report.py categories` for the category attribution of every overall-board handle, and the
`handle_metrics` all-time series differenced against market expiry times in `markets` (CPI, FOMC,
game starts) to find event-locked earners.

## 7. Open items
- Meaning of projected_pnl (unrealized? window?).
- Which categories the leaderboard filter accepts beyond Sports.
- Depth of history on `social/trades` for a Full user (paged only 18 minutes today).
- Website Terms of Use and Developer Agreement text.
- The Bloomberg inflation-trader piece.
