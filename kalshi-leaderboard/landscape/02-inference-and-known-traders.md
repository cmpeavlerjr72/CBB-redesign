# 02 - Inference from public tape, and what is publicly known about top Kalshi traders

Date: 2026-11-02. Public endpoints only. Tags: [F] fetched and read this session, [S] search-snippet only (UNVERIFIED), [U] UNVERIFIED/not fetched.
Raw pulls: C:\Users\devuser\kalshi-leaderboard\samples\inference\ (untrusted). Scripts: ...\scripts\inf_pull.py, inf_pull2.py, inf_analyze.py (run with python -I).

## 1. Summary

- Public tape (GET /markets/trades) has ticker, count_fp, yes/no price, taker_side, taker_book_side, created_time, trade_id (a per-print UUID, not a user id), is_block_trade. No account, order id, or maker id. [F, samples]
- Individual traders cannot be identified from the tape. Strategy CLASSES can be partly inferred (algorithmic slicing, settlement-time 1c sweeps, fixed-dollar retail tickets, resting whale bids), and a few can be tracked as pseudo-identities within one market over minutes. They cannot be linked to a leaderboard handle.
- Kalshi leaderboard (from the other worker's sample lb_pnl_all_time.json) gives only nickname, value, rank; NO market category. Top all-time entries there: 4thewin $7.74M, final.lamb3899 $7.33M, dogname $6.54M, user.x $5.71M, COBYBETS1onX $4.56M, Flupnolide $4.43M, mrred $3.96M. Category per handle is therefore unknown from Kalshi itself. Leaderboard opt-in only (users opted out by default) [F help.kalshi.com/en/articles/13823809-leaderboard].
- Third-party sites (kalshispy, rivo, predicting.top) claim to attach handles to specialties and per-trade positions; their numbers do not match Kalshi's own board (kalshispy top is $963K vs $7.7M on the Kalshi sample), so they cover only a subset or use a different window/definition. Rivo says names come from Kalshi public profiles and it scores trades itself at settlement [F rivo whale tracker page]; how it links trades to profiles is not explained (UNVERIFIED; whether Kalshi exposes per-user data is the other worker's question).
- Press coverage naming individual top Kalshi traders with verified methods is thin. I found NO Bloomberg/WSJ/NYT/Sportico profile I could actually read: a Bloomberg feature "Kalshi's Top Inflation Trader Beats Wall Street Banks From His Garage" (2026-10-13) exists per search listing but returned HTTP 403 [S]. It is the only named lead for the econ category.
- Sports is not the only place to look, but the evidence base for the other categories is mostly vendor blogs. Hard evidence on edge exists for econ (Fed paper) and adverse selection (Bartlett-O'Hara); weather and crypto evidence is mostly self-reported or skeptical.

## 2. Known traders and methods (by category)

Caveat: scale numbers are third-party lifetime realized PnL, not Kalshi-verified.

### 2a. Named handles and institutions

| Name / handle | Source | Claimed strategy | Scale | Tag |
|---|---|---|---|---|
| Susquehanna (SIG), "Susquehanna Predictions" | Kalshi/BusinessWire Apr 2024; Tarek Mansour X post | First committed institutional MM; continuous two-way quotes; reduced fees and higher limits per Turbine; "~30x liquidity" in select markets | Not public | [S] snippets of businesswire.com/news/home/20240403664852; kalshi.com/blog/article/kalshi-kit-liquidity-sig-market-makers |
| Jump Trading | Bloomberg via blogs | Equity stakes in Kalshi and Polymarket Feb 2026 in exchange for liquidity provision; event-contract team ~20 | Not public | [S] secondary only |
| "2,000+ small market makers, ~95% of bid matches" | Founder interview, cited by Turbine blog | Long tail of retail/bot makers | n/a | [F turbinefi blog] but claim itself UNVERIFIED (one interview) |
| @BWmarkets | kalshispy, rivo | Sports (baseball leader on rivo); 284 markets, ROI 11.7% | +$810K to +$963K lifetime (two sites disagree) | [F kalshispy], [S rivo] |
| @Unitdor | kalshispy | Sports, 24,212 markets (high count = many small positions, MM-like) | +$589K | [F] |
| @BitcoinTradingChallenge | kalshispy | Sports, 32,134 markets, ROI 0.7% (volume strategy, thin margin: MM or arb signature) | +$488K | [F] |
| @autofade | kalshispy | Sports, 13,212 markets | +$474K | [F] |
| @locksy | kalshispy | Mentions, 12,556 markets | +$294K | [F] |
| @szg.szg | kalshispy | Crypto, 10,656 markets, ROI 0.2% | +$287K | [F] |
| @timmy, @bullishkirby | kalshispy | Elections, few markets (65, 37), concentrated directional bets | +$501K, +$266K | [F] |
| @Domahhhh ($980K), @GaetenD ($420K), @Foster (mentions), @cobybets1 (sports/politics), @debl00b ($42M volume), @PredMTrader | Auli X post, Oct 2025 | Self-promoting follow-list | as stated | [S] snippet of x.com/aulijk/status/1977453499131797531; Domahhhh has a podcast appearance on finding edge (description only) |
| Kalshi leaderboard top: 4thewin, final.lamb3899, dogname, user.x, COBYBETS1onX, Flupnolide, mrred | Kalshi board sample (other worker) | None disclosed | $4M-$7.7M all-time | [F sample file] |
| Michael Ashton ("Inflation Guy", ex-Barclays) | Bloomberg via advisorperspectives, Oct 23 2026 | Designing Kalshi CPI futures (planned Dec 2026). Not a leaderboard trader. The Bloomberg "top inflation trader" feature is a different, unnamed person | n/a | [S] |
| Polymarket "Theo" / Fredi9999 | WSJ, Bloomberg, Chainalysis, Nov 2024 | Directional Trump win bet, French ex-trader, commissioned own polls ("neighbor" poll) | $48M to $85M across 4 to 11 accounts (Chainalysis said 9 accounts, ~$79M) | [S] bnnbloomberg, theblock |

### 2b. By category (what is reported, claimed edge, what tape would show)

| Category | Who is reported big | Claimed edge | Public tape signature | Tag |
|---|---|---|---|---|
| Sports (game winners, props) | BWmarkets, Unitdor, autofade, BitcoinTradingChallenge; SIG as MM; sports >90% of volume (Wikipedia) | MM spread capture; Bartlett-O'Hara: makers earn 1.91c/contract on single-name vs 0.82c broad-based, won 63.4% of single-name contracts; in-play latency (but one NFL ESPN-feed bot failed because pro feeds are "at least 30 seconds faster" [F hackingthemarkets]) | Heavy tape; resting-size repeats, burst after scoring; settlement-time sweeps (Section 4) | [S] for stats, [F] for bot post |
| Politics / election night | @timmy, @bullishkirby (elections); cobybets1 (sports+politics); Polymarket whales | Directional conviction; Kalshi-Polymarket arb. Guides say liquid markets converge within 1-2% and gross 3c edges vanish after fees; resolution-wording risk | Large single prints, thin trade counts. Arb legs show as near-simultaneous fills and Kalshi price lead/lag vs Polymarket | [S] vendor blogs only; no named arb trader found |
| Economic data (CPI, Fed, jobs, GDP) | No verified named top trader (Bloomberg garage-trader story not readable) | Fed Board FEDS paper: Kalshi beats Bloomberg consensus on headline CPI and beats fed funds futures; edge claimed in tail brackets and from nowcasts/release-time speed (vendor claims) | Release-time bursts at the exact BLS/FOMC second; stale resting quotes picked off (one-sided taker flow in the first seconds); MM quote withdrawal before release | [S] FEDS 2026010pap.pdf; blogs are commercial |
| Weather / climate | None named | NWS model vs bracket price; skeptical evidence: 1,506 NYC high-temp events backtest found crowd about as accurate as NWS model after fees; competing bots allegedly react within seconds of model cycle updates; same-day markets often already reflect observed highs | Price steps clustered after NWS/model update times (e.g., 00/06/12/18Z cycles) and METAR observations; fixed bracket laddering | [S] oddsreference, northlakelabs, cwendt6 repo - self-reports |
| Mentions / culture / entertainment | @locksy (12.6K markets), @Foster ("best mentions trader" per X post) | Transcript base rates; CFTC probing mention markets, Kalshi removed sports mention markets (NPR Aug 2026) | Many small multi-word contracts per event, spikes during live speech (word said = price jump to ~99c) | [S] NPR, defirate |
| Crypto price (15-min, hourly) | @szg.szg (ROI 0.2%) | Spot-feed latency (Binance/CF Benchmarks) and MM; Synth (via news report) says Kalshi 15-min BTC moves predict Binance; Polymarket study median 347 ms Binance lead | Sub-second bursts aligned to spot ticks; both-side buying when yes+no ask < $1 (bot repos) | [S] secondary, Synth study not found |

## 3. Inference methods

| # | Method | Data needed | Yields | Confidence |
|---|---|---|---|---|
| 1 | Pseudo-identity clustering by exact size/price/timing | Trades (count_fp, price, created_time); WS `trade` | Repeated exact sizes with sub-100ms gaps = one algorithm slicing/sweeping (see 935.19 below). Does NOT give identity, and cannot join across markets or days reliably | Medium for same-market minutes; low across days |
| 2 | Maker vs taker split | taker_side, taker_book_side (present in tape) | Direction of the aggressor; the resting side is the counterparty but anonymous. Per-size resting-bid detection needs orderbook_delta | High for aggressor side; makers inferred only |
| 3 | Markout / CLV by flow class (size bucket, market type) | Trades + later mid/settlement | Whether large taker prints win more than small ones (informed vs retail); compare maker-side PnL = inverse of taker. Feasible fully offline on public data. Matches Bartlett-O'Hara method | High for aggregate classes; none for individuals |
| 4 | Latency trader detection in-play | Trades + ESPN/league event timestamps | Burst latency distribution after scoring events; reveals speed class, not who | Medium; event timestamp quality limits it |
| 5 | Cross-venue lead/lag | Kalshi trades/quotes + Polymarket CLOB + Pinnacle | Which venue moves first; arb sizing. Arb traders look like two near-simultaneous prints, not identifiable | Medium |
| 6 | Settlement sweeps | Trades in the last minutes / after outcome known | 1c / 99c sweeps by bots (in sample below, 203 identical 935.19-lot prints in 22 distinct seconds) | High that they exist; low on who |
| 7 | Orderbook_delta reconstruction | WS orderbook_delta with sequence | Order add/cancel timing, queue refresh cadence, size ladders: strongest pseudo-identity signal for makers (e.g. 12.1c x 23k resting bid in the user's own notes). Needs a live collector, not retroactive | Medium-high for stable resting patterns |
| 8 | Order-flow toxicity (VPIN, Easley, Lopez de Prado, O'Hara 2012), footprint charts, crypto whale clustering | Volume buckets | Equity/crypto analogs: aggregate toxicity and address clustering. On Kalshi no address-level link exists, so only the aggregate VPIN-style part transfers | Brief reference; [U] not re-verified |

Related own-research hooks (not copied): project_0728_strategy_reckoning.md records that sub-floor sell dumps (23,346ct Suzuki-HR-repack dump priced 0.9c under floor) were caught by "someone's standing 12.1c deep bid" ("a pro already harvests this"), i.e., a resting-bid whale pattern already seen via RFQ data; and that pregame combo flow is bound-clean. reference_kalshi_api_full_map.md lists /incentive_programs (Kalshi pays makers for size), `user_orders` WS, and /live_data game_stats as unused surfaces; the incentive programs are the public route to understanding why large resting size exists at all.

## 4. Experiment (verified, run 2026-11-02)

Source: GET https://api.elections.kalshi.com/trade-api/v2/markets/trades?ticker=...&limit=1000 with cursor (and min_ts/max_ts for windows). Tickers found via /markets?series_ticker=KXNFLGAME|KXMLBGAME&status=settled. All four are settled October/November 2026 game-winner markets. "Window" runs from 30 min before market close time (close_time is when the winner was declared, so the window is the late game / resolution period) except row 1, which is the full market life. Size unit: count_fp contracts (fractional).

| Market | Window | Prints | Contracts | Median / mean size | Top 1% prints share of volume | Prints >=1000 (share of volume) | Taker YES share of volume | Max prints in one second | Most repeated sizes (count) |
|---|---|---|---|---|---|---|---|---|---|
| KXNFLGAME-26OCT25KCSEA-KC | full life (~12.8 days) | 240,303 | 47.18M | 36.6 / 196 | 43.1% | 6,831 (56.8%) | 81.2% | 624 | 22.88 (9,901), 11.44 (9,592), 45.76, 57.2, 114.4 |
| KXNFLGAME-26NOV01KCDEN-KC | last 30 min | 57,454 | 12.58M | 30.2 / 219 | 46.3% | 1,883 (63.1%) | 65.8% | 376 | 1.0 (1,780), 10, 5, 40, 13.66; big: 935.19 (203), 2000 (104) |
| KXNFLGAME-26NOV01MINDET-MIN | last 30 min | 15,190 | 2.92M | 32.9 / 192 | 40.1% | 445 (56.7%) | 56.8% | 96 | 1.0, 25.22, 15.88; big: 935.19 (46), 2000 (26) |
| KXMLBGAME-26OCT282000MILTB-MIL | last 30 min | 5,205 | 1.38M | 16.1 / 265 | 58.9% | 188 (76.2%) | 32.2% | 587 | 5.04, 1.0, 10.09, 17.28; big: 500 (17), 504.69 (10) |

Observations
- Concentration: 1% of prints carry 40% to 59% of contracts; prints >=1000 contracts carry 57% to 76%. Median print is only 16 to 37 contracts. So volume is dominated by a small number of large orders, yet at print level it is retail-sized.
- Fractional repeated sizes (22.88, 11.44, 45.76, 5.04, 10.09, 17.28): consistent with fixed-dollar tickets converted to contracts at a price (e.g. $10 / 0.437 = 22.88). Those are app-style/retail dollar-sized orders (inference, not confirmed).
- KC-DEN repeated size 935.19: 203 prints, all taker YES at 0.01 yes price (taker_book_side bid), median gap 56 ms, 10th percentile 12 ms, spread over only 22 distinct seconds. That is one automated actor sweeping 1c in the final minutes (KC YES settled NO, so these are post-outcome 1c lottery/sweep or position-clearing prints; why they buy YES at 1c is not explained by the tape). Size 2000 appears 104 times (81 YES-taker, 23 NO-taker, prices dispersed 0.28 to 0.42, gaps median 3 s): looks like a round-lot algo working an order across prices, or several actors using the same round size. Cannot tell which.
- Same 935.19 lot also appears 46 times in MIN-DET at the same session end, suggesting the same actor across markets at the same time (cross-market pseudo-identity, plausible but unproven).
- Taker imbalance: YES-taker share is 57% to 81% in the NFL markets, 32% in the MLB market; the full-life KC-SEA 81% YES-taker with result NO is the classic favorite/longshot pattern (takers buying YES that lost; makers on the NO side profit), consistent with the Kalshi maker/taker literature (Bartlett-O'Hara; Burgi-Deng-Whelan CESifo) [S].
- Bursts: p99 prints per active second is 17 to 200; peak 376 to 624 prints in one second at resolution. Volume per minute near the close is ~1.8M to 2M contracts in the last two minutes of KC-DEN vs 80K in the first minute of the window: heavy end-of-market concentration.
- is_block_trade was False for all KC-DEN prints. trade_id is a UUID with no user linkage.

What the tape does and does not reveal
- Reveals: size distribution, aggressor side, timing microstructure, evidence of algorithmic slicing and sweeps, and aggregate informed-vs-uninformed flow.
- Does not reveal: any account, maker identity (the resting side is unlabeled), P&L, positions, cross-market identity, or what a given leaderboard handle did. Matching a handle to a print would need either an exact size/time fingerprint known from outside (e.g., your own fills) or non-public data.
- Limits: one 30-minute window per market except one full pull; October sample; no orderbook_delta captured (the strongest maker signal needs live WS collection); no markout computed yet.

## 5. Assessment: what "copying" a leaderboard trader can realistically mean

- Trade mirroring (copy their positions): not possible from public data. No per-user trade list, no handle on prints, and by the time anything is attributable (e.g. a third-party site scoring a trader at settlement) the edge is gone. Whale-tracking on Polymarket works only because wallets are on-chain; the Kalshi equivalent does not exist publicly. Third-party "Kalshi whale trackers" (rivo, kalshispy) claim handle-linked positions; mechanism unexplained, treat as UNVERIFIED until the other worker's findings land. Copy-trading bots exist for Polymarket (Kreo, PolyCop, GodEye, Polywhaler [S]); sources warn copying is backward looking and survivorship-biased. I found no admitted Kalshi copy-trading.
- Strategy-class imitation: realistic and the right target. Handle metadata gives a hint at class: count of markets traded (for example Unitdor 24K, BitcoinTradingChallenge 32K, szg.szg 10.7K markets at ROI 0.2% to 0.7% imply volume/MM or arb; BWmarkets 284 markets at 11.7% ROI, timmy 65 markets, devex 15 markets at 40% ROI imply concentrated directional/informational bets). That distinction, volume-per-profit, is the cheapest public signal of "market maker vs bettor".
- For a market-making operator the tape gives usable inputs: flow toxicity by size bucket and market, expected maker edge from taker imbalance, settlement-sweep timing, and where large resting bids sit (needs orderbook_delta). Those can be measured on your own bot's fills plus public tape without knowing who anyone is.
- Category caution: the coordinator's hypothesis that top earners are not sports/RFQ is not supported or refuted here. Kalshi's own board has no category field; third-party tags put 6 of the top 10 on kalshispy in sports, but that list is a different, smaller population than the Kalshi board. Resolving it needs the other worker's per-handle data, e.g. num-markets-traded and any profile markets list.

## 6. Links

- Kalshi leaderboard help: https://help.kalshi.com/en/articles/13823809-leaderboard [F]
- kalshispy top traders: https://kalshispy.com/top-kalshi-traders [F]; rivo leaderboard https://rivo.markets/kalshi-leaderboard , whale tracker https://rivo.markets/kalshi-whale-tracker [F]
- Turbine on adverse selection / MMs: https://www.turbinefi.com/blog/why-prediction-market-trades-get-picked-off-2026 [F]
- SIG MM announcement: https://www.businesswire.com/news/home/20240403664852/en/Kalshi-Onboards-Its-First-Dedicated-Institutional-Market-Maker ; https://x.com/mansourtarek_/status/1775542217307791366 [S]
- Bartlett and O'Hara, Adverse Selection in Prediction Markets: Evidence from Kalshi (NBER draft) https://conference.nber.org/conf_papers/f243677.pdf [S]; Burgi, Deng, Whelan, Makers and Takers: https://www.ifo.de/en/cesifo/publications/2026/working-paper/makers-and-takers-economics-kalshi-prediction-market [S]
- Fed Board econ-market paper: https://www.federalreserve.gov/econres/feds/files/2026010pap.pdf [S]
- Bloomberg inflation-trader feature (403): https://www.bloomberg.com/news/features/2026-10-13/kalshi-s-top-inflation-trader-beats-wall-street-banks-from-his-garage [S, not fetched]
- NPR mention markets probe: https://www.npr.org/2026/08/13/nx-s1-5930689/cftc-probe-mention-markets-prediction-markets-kalshi [S]
- Weather: https://oddsreference.com/predictions/research/kalshi-weather-model ; https://www.northlakelabs.com/max/blog/kalshi-weather-postmortem-and-pivot/ [S]
- ESPN-feed NFL bot post: https://hackingthemarkets.com/anatomy-of-a-kalshi-nfl-trading-bot/ [F, paywalled partially]
- Polymarket Theo: https://bnnbloomberg.ca/investing/2024/11/07/new-analysis-boosts-trump-whales-polymarket-haul-to-85-million [S]; copy-trading tool survey https://dev.to/itsragnar/best-polymarket-trading-tools-in-2026-analytics-copy-trading-execution-arbitrage-bots-e43 [S]
- Auli follow-list: https://x.com/aulijk/status/1977453499131797531 [S]
- Not found / blocked: Reddit r/Kalshi and r/predictionmarkets threads (not searched successfully; Reddit API is closed to the user), WSJ/NYT/Sportico/Front Office Sports profiles of named Kalshi traders, any admitted Kalshi copy-trading, any Kalshi-published list of MM firms beyond SIG and Jump.
