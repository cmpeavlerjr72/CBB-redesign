# What Kalshi exposes about trader identity (fetched 2026-11-02)

All endpoints below were fetched unauthenticated with browser-like headers at <=2 req/s. Samples: `samples/kalshi/` (untrusted); scripts: `scripts/fetch.py`, `scripts/sweep_trades.py`.
Base for the social service: `https://api.elections.kalshi.com/v1` (found as `exchangeApiUrl` + `/v1` in the Next.js bundle; NOT under /trade-api/v2).

## 1. Summary

| Question | Answer |
|---|---|
| Handles on leaderboard | YES, public, no login (API). The web page /social/leaderboard also renders a partial SSR version without login. |
| Profit and volume per handle | YES (leaderboard values, plus `profile/metrics` for any nickname) |
| Positions per handle | NO without login (`social/profile/positions` is a POST; client code throws "Profile positions require authentication"; GET 404s). UNVERIFIED what a logged-in viewer sees (states content/empty/hidden/request/requested). Not tested (no login allowed). |
| Trades per handle | YES for users on the "Full" sharing level: `GET /v1/social/trades?nickname=X` returns their fills, paginated, unauthenticated. Limited/Minimum users return `{"trades":[],"visibility_state":"hidden"}` (leaderboard #1 weekly final.lamb3899 is hidden). |
| Handles on market-level trades | PARTIAL. Kalshi's social trade feed carries `maker_nickname`/`taker_nickname`; about 9% of trades (91 of 1000 sampled) had one filled. The official trade-api v2 `/markets/trades` has NO identity (confirmed). |
| Historical leaderboard snapshots | NO. Only live windows (daily/weekly/monthly/yearly/all_time). No snapshot endpoint found; we must snapshot ourselves. |

Bottom line: Full sharing is a deliberate public activity feed, so tracking works for the subset who chose Full. Many top-PnL names are Limited and invisible at trade level.

## 2. Endpoint catalog

All GET, no auth, JSON, unless noted.

### 2.1 Leaderboard
`GET /v1/social/leaderboard?metric_name=<m>&limit=<n>&time_period=<p>[&category=<c>]`
- metric_name: `projected_pnl`, `volume`, `num_markets_traded` (UI: Profit / Volume / Predictions). `metric_pnl` is rejected (400 oneof).
- limit: max 60 (limit=500 gives 400 "max"). No cursor, so top 60 only.
- time_period: `daily|weekly|monthly|yearly|all_time` (from UI bundle; yearly and all_time fetched OK). Daily and weekly returned identical rows in my test, so one may be an alias or a bug: UNVERIFIED. Omitting the param behaved like weekly.
- category: UI list Politics, Sports, Culture, Crypto, Climate, Economics, Mentions, Financials, Science, Elections (+ perps). `category=Sports` changed the ranking. Other spellings UNVERIFIED.
- Response: `{"rank_list":[{nickname, social_id (always "" here), profile_image_path, value, rank, is_anonymous}], "coverage":{provisional,covered_users,cutoff_ms}}`. value: PnL in dollars (float), volume in contracts, count of markets.
- Cadence: UI text "The data is updated approximately every 5 minutes." Client refetches every 60 s.
- Own rank: `GET /v1/users/{userId}/social/leaderboard_rank` (auth).
Samples: lb_projected_pnl_weekly.json, lb_pnl_monthly.json, lb_pnl_all_time.json, lb_volume_weekly.json, lb_num_markets_traded_weekly.json, lb_pnl_weekly_sports.json.

### 2.2 Profile
`GET /v1/social/profile?nickname=X` returns `social_profile{social_id (UUID), nickname, follower_count, following_count, posts_count, profile_image_path, description, joined_at (date), profile_view_count, top_categories[]}` and `inner_circle{enabled,viewer_status}`.
`GET /v1/social/profile/metrics?nickname=X&time_period=P` returns for Full users `metrics{volume, dollars_traded, pnl, num_markets_traded}` (pnl appears to be in 1e-4 dollars: 619709248 for final.lamb3899 weekly vs 61970.9 on the leaderboard); a user with hidden trades (clis41) returned only `open_interest`. Also `social_id`.
`GET /v1/social/profile/category_metric` requires a `metric` param (values UNVERIFIED); `profile/subcategory_metrics` seen in bundle, not fetched.
Authenticated variants (`users/{uid}/social/profile_view...`) exist; not tested.
Web profile URL pattern: UNVERIFIED.

### 2.3 Per-user and global trade feed (the crux)
`GET /v1/social/trades?page_size<=100&cursor=<opaque>&nickname=X&min_amount&series_ticker&event_ticker&market_ticker`
Row: `trade_id (UUID), market_id, ticker, price, price_dollars, count, count_fp, taker_side, taker_action (buy/sell), maker_nickname, taker_nickname, create_date`. Response also `cursor`, `visibility_state`.
- Without nickname: global recent-trades feed. 1000 rows spanned about 5 s of market time (very high volume); 91 had nicknames. A nickname appears on whichever side is a Full user.
- With nickname=clis41 (Full): 600 trades over 6 pages going back ~18 minutes via cursor. Maximum history depth UNVERIFIED (did not page to exhaustion).
- Supports filtering by market_ticker/event_ticker/series_ticker.
- Hidden users: `{"trades":[],"visibility_state":"hidden"}`.
- nickname is mutable (`social/update_nickname` exists); `social_id` UUID (from /social/profile) is the stable key.
Samples: social_trades_public.json, social_trades_sweep.json, clis41_trades.json, social_trades_nick.json (hidden).

### 2.4 Positions and holders
- `POST /v1/social/profile/positions {nickname, limit, closed_positions, cursor}` returns `{state, event_positions[], cursor}`. Client requires login. Unauthenticated POST not attempted: UNVERIFIED. Positions can be reconstructed from the fill feed instead.
- `GET /v1/social/margin/top_holders?market_ticker=T` returns `rank_list[{nickname, social_id, profile_image_path, value, rank, is_anonymous, side}]`. Seems to be for perp/margin markets: 404 market_not_found on two regular event tickers. UNVERIFIED on a perp ticker.
- `GET /v1/social/notable` 404. A "Notable account" green-ring badge exists in the UI; field name not seen.
- Posts/comments: write endpoints `POST users/*/social/posts` seen; public read UNVERIFIED (GET /v1/social/posts 404).

### 2.5 Sharing preferences (privacy model, from bundle)
Set per user via `POST users/{id}/social/sharing_preference`; three levels:
- Full: "Your entire profile, including trading activity". Eligible for all leaderboards.
- Limited: "Your entire profile, but your trading activity stays hidden". Eligible for social leaderboards.
- Minimum: "Posts and followers only." Hidden from all leaderboards.
Help Center https://help.kalshi.com/en/articles/13823809-leaderboard: "While you're automatically opted out of the leaderboard by default, you have the flexibility to opt in at any time."
Also `social/inner_circle/*` (request/accept/members), a follower-approval tier.

### 2.6 Official trade API (identity-free)
- `GET /trade-api/v2/markets/trades` (unauth): fields count_fp, created_time, is_block_trade, no_price_dollars, taker_book_side, taker_outcome_side, taker_side, ticker, trade_id, yes_price_dollars. No user id. CONFIRMED.
- `GET /trade-api/v2/markets/{ticker}/orderbook` (unauth): price levels only.
- `GET /trade-api/v2/communications/rfqs` returns 401 without a key.
- WebSocket `trade` channel (https://docs.kalshi.com/websockets/public-trades.md): trade_id, market_ticker, price, count, taker_side, taker_outcome_side, taker_book_side, ts. No user id. The WS connection itself needs API-key auth.
- `fills`, `market_positions`, `user_orders` channels are private.

### 2.7 Copy trading is a first-party feature
Bundle has `POST /users/{uid}/social/posts/{postId}/copy_trade` and `POST /live_chat/copy_trade`: Kalshi's own one-click copy from a social post or live chat.

### 2.8 RFQ identifiers (coordinator question)
Docs https://docs.kalshi.com/websockets/communications.md and https://docs.kalshi.com/api-reference/communications/get-rfqs.md:
- `creator_id`: "Public communications ID of the RFQ creator (pseudonymous). When obscure_creator_id is enabled, other users receive the shared placeholder "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" (SHA-256 of empty input). The creator sees their normal ID. After successful quote execution, subsequent RFQ responses show the normal public communications ID to all viewers." The WS doc example is 64 hex ("43fc3733603f...b63a51").
- Quote messages carry `quote_creator_id` and `rfq_creator_id`; batch child quotes use the shared placeholder.
- REST RFQ also has `creator_user_id` ("User ID of the RFQ creator (private field)"); filter param `creator_user_id` is deprecated and takes a UUID.
- So obscuring exists as an option (obscure_creator_id), and the ID becomes visible after a quote executes. The trader should check whether their own RFQs use it.

Identifier formats and linkability:

| identifier | where | format | linkable? |
|---|---|---|---|
| nickname | leaderboard, social trades, profile | string, auto "adjective.animal####" or custom | mutable; join key across social endpoints |
| social_id | /social/profile, top_holders | UUID (32420684-158d-459c-8c03-9ffd4e85adc9) | stable, public; "" on leaderboard rows |
| profile_image_path | leaderboard | string embedding nickname + epoch ms, or pool path | no |
| trade_id | social trades and trade-api v2 | UUID, same format in both | probably the same ID space, so a nickname could be tied to an official v2/WS trade; exact match across both feeds not demonstrated (samples taken seconds apart): UNVERIFIED |
| RFQ creator_id / quote_creator_id | WS RFQ/quote messages | 64 hex | NO 64-hex id appears in any social endpoint or the public trade WS channel. No public mapping to nickname or social_id found. Mapping to the UUID would need the (unknown) derivation: UNVERIFIED |
| creator_user_id | REST RFQ, private | UUID | possibly equals social_id; untestable without auth |

## 3. Leaderboard top rows, 2026-11-02 (API, no login)

**pnl weekly (projected_pnl, time_period=weekly)** (60 rows; daily returned the same rows)

| rank | nickname | value |
|---|---|---|
| 1 | final.lamb3899 | $61971 |
| 2 | fluffy.mole7725 | $49104 |
| 3 | married.sloth4281 | $38558 |
| 4 | visiting.wolverine9239 | $36718 |
| 5 | TheHandOfTheKing | $28542 |
| 6 | ripe.swordtail3199 | $22591 |
| 7 | theoretical.rattlesnake9798 | $22438 |
| 8 | genetic.silkworm8197 | $20023 |
| 9 | lostzemblan | $18245 |
| 10 | tasty.salmon8122 | $15716 |

**pnl monthly** (60 rows)

| rank | nickname | value |
|---|---|---|
| 1 | awake.goldfish3699 | $341737 |
| 2 | likely.corgi1522 | $301316 |
| 3 | youie | $295205 |
| 4 | jackmack | $213079 |
| 5 | ripe.swordtail3199 | $190768 |
| 6 | married.sloth4281 | $171682 |
| 7 | longcastles | $160025 |
| 8 | gocatsnu | $140910 |
| 9 | lazydog | $130636 |
| 10 | Bizzle333 | $120923 |

**pnl all_time** (60 rows)

| rank | nickname | value |
|---|---|---|
| 1 | 4thewin | $7742386 |
| 2 | final.lamb3899 | $7326914 |
| 3 | dogname | $6537846 |
| 4 | user.x | $5710630 |
| 5 | COBYBETS1onX | $4556710 |
| 6 | Flupnolide | $4434323 |
| 7 | mrred | $3962109 |
| 8 | valuable.loon4613 | $3590432 |
| 9 | ClayA | $3029966 |
| 10 | longcastles | $2957243 |

**volume weekly** (60 rows)

| rank | nickname | value |
|---|---|---|
| 1 | wonderful.cougar7351 | 31,070,204 contracts |
| 2 | mechanical.antelope0613 | 4,881,199 |
| 3 | 4thewin2.0 | 4,696,303 |
| 4 | DoklerPredicts | 4,510,863 |
| 5 | written.sheepdog2191 | 3,576,015 |
| 6 | creepy.fish8178 | 3,198,256 |
| 7 | sleek.duck0255 | 3,181,589 |
| 8 | reaviat | 3,053,495 |
| 9 | sovoks | 2,773,252 |
| 10 | final.lamb3899 | 2,623,245 |

**num_markets_traded weekly**: 1 wonderful.cougar7351 19,861; 2 surviving.raven1227 3,057; 3 adverse.lemur1301 2,484; 4 .bananafish 2,117; 5 objective.emu1869 2,037.

Notes: the SSR HTML at https://kalshi.com/social/leaderboard (weekly, all categories) showed a different, sparse sample (ranks 7, 11, 12...: BaronVonBid $96,979; quiet.badger5137; soupwins...) from the API weekly list, so the SSR payload is stale or filtered; use the API. `/leaderboard` and `/leaderboards` 404; the page is /social/leaderboard (tabs Leaderboard / Activity). The web UI shows a login gate copy ("Unlock the leaderboard"), but the API is not gated. Many top names are auto-generated (adjective.animal####): stable per account but no real identity.

## 4. Terms
- Member Agreement (https://kalshi.com/docs/kalshi-member-agreement.pdf, fetched, samples/kalshi/ma.txt): NO clause on scraping, bots, or copy trading (searched scrap/robot/spider/automat/copy trad/API/social/leaderboard). Section XI Data Use Consent: "You hereby grant Kalshi a worldwide, perpetual, irrevocable, royalty-free, full sublicensable and freely assignable license to store, use, copy, display, disseminate and create derivative works from: (i) the price and quantity data for each transaction entered into by You that is executed via the Services and (ii) each bid, offer and/or order provided via the Services by You."
- Privacy Policy (https://kalshi.com/docs/kalshi-privacy-policy.html): "Profile Data includes your username and password, trades or orders made by you, and your interests..." Nothing found on leaderboard publication.
- Website Terms of Use (https://kalshi.com/regulatory/terms-of-use) and API/Developer Agreement: UNVERIFIED. Client-rendered; WebFetch got HTTP 429. This is where an anti-scraping clause would most likely live. Read manually before building a collector.
- Leaderboard and trade visibility are user opt-in (Help Center quote in 2.5).

## 5. Third-party trackers (WebSearch; only two pages actually read)

| Tool | URL | Claims | Method | Status |
|---|---|---|---|---|
| Kalshi Leaderboard Scraper (Apify) | https://apify.com/saswave/kalshi-leaderboard-scraper | leaderboard by volume/profit/predictions, categories, timeframes, ~$0.50/1000 results | not stated | exists per search; direct fetch 404 |
| Bot for Kalshi whale tracker | https://www.botforkalshi.com/tools/kalshi-whale-tracker | largest recent fills grouped by market/side with VWAP; no trader names | "Kalshi's live trade feed" | fetched; unaffiliated |
| Oddpool whales | https://www.oddpool.com/whales | Kalshi + Polymarket trades >= $1,000, API | "we watch order book filling" (Kalshi) | fetched; no handles stated |
| Predicting.top | https://pm.wiki/ja/projects/predicting-top | leaderboard of top Polymarket and Kalshi traders by PnL/win rate/volume | not stated | UNVERIFIED (listing only) |
| 0xInsider | https://www.trevorlasn.com/blog/0xinsider-prediction-market-whale-tracking | whale trade tracker, both platforms, beta | not stated | UNVERIFIED |
| PolyTrack | https://alternativeto.net/software/polytrack/about | Polymarket whales | on-chain | Polymarket only |

No GitHub repo using /v1/social/trades was found; X bots and Discord not searched in depth. The nickname-filtered trade feed looks under-exploited.

## 6. Polymarket contrast (general knowledge, UNVERIFIED today)
Polymarket positions and trades are on-chain keyed by wallet, so every account's full history is public by default (Dune, analytics sites, copy-trading bots). Kalshi is custodial: no per-user history in the official API; identity exists only through the voluntary social layer (Full users get a public nickname-keyed fill feed; leaderboard capped at 60 rows). Limited/Minimum users, including some top-PnL names, are invisible at trade level.

## 7. Implications for the bot
- Poll `/v1/social/leaderboard` (metrics x windows x categories, every ~5 min) and store snapshots to create the history Kalshi lacks.
- Per leaderboard nickname: `social/profile` once for social_id, then `social/trades?nickname=` incrementally by cursor for Full users; track hidden rate.
- Gives price, size, side, time, ticker; not resting orders or intent. Reconstruct positions by summing fills.
- Outstanding: Terms of Use and Developer Agreement review.

## 8. Links
- https://kalshi.com/social/leaderboard
- https://api.elections.kalshi.com/v1/social/leaderboard , /v1/social/trades , /v1/social/profile , /v1/social/profile/metrics
- https://help.kalshi.com/en/articles/13823809-leaderboard
- https://docs.kalshi.com/llms.txt , /websockets/public-trades.md , /websockets/communications.md , /api-reference/communications/get-rfqs.md
- https://kalshi.com/docs/kalshi-member-agreement.pdf , https://kalshi.com/docs/kalshi-privacy-policy.html

## 9. Top-handle visibility sweep (2026-11-02)

Method: union of top 60 by projected_pnl monthly, projected_pnl all_time and volume all_time = 141 distinct handles. For each: `/v1/social/profile`, `/v1/social/profile/metrics` (all_time and monthly), `/v1/social/trades?nickname=&page_size=100` (up to 2 pages = max 200 fills). Raw JSON per handle: `samples/kalshi/handles/<nick>.json`; table: `samples/kalshi/top_handles_sweep.csv`; Full-handle series mix: `samples/kalshi/top_handles_series_mix.md`; script `scripts/handle_sweep.py`. Classification: Full = trade feed returned fills; Limited = trades hidden but pnl metrics returned; Minimum/hidden = trades hidden and no pnl metrics.

Counts (141 handles): Full 5, Limited 112, Minimum/hidden 24.
- Monthly PnL top 60: 4 Full (far.bike #17, Lajollawinner #23, szg.szg #42, BSharp #49), 56 Limited.
- All-time PnL top 60: 1 Full (BWmarkets #60), 59 Limited. So 98% of the all-time top earners, and 93% of the monthly top earners, are NOT trackable at trade level.
- Volume all_time top 60: 1 Full (szg.szg #49), 35 Limited, 24 Minimum/hidden (these show on the volume board with no PnL metrics, including weekly volume #1 wonderful.cougar7351; this contradicts the bundle text that Minimum is hidden from all leaderboards, so the Minimum label is only inferred).
- Visibility correlates inversely with success: the biggest earners overwhelmingly picked Limited.

Caveats: fills limited to the latest 200 per handle, so windows are short (1 hour to 8 days). far.bike's all-time pnl metric (-$91k) is inconsistent with its monthly rank of 17: the metric/unit semantics (pnl field looks like 1e-4 dollars) or window handling is UNVERIFIED. Positions were not fetched.

Categories traded by the 5 visible handles (share of the 200 fills):
- Sports 4 of 5 (far.bike, Lajollawinner: cross-category parlay series KXMVECROSSCATEGORY 79% / 100%; BSharp: esports and college hoops/NBA/NFL spreads and totals; BWmarkets: college football totals and spreads).
- Crypto 1 of 5 (szg.szg, 100% KXBTC15M, the 15-minute BTC contracts).
- Politics, econ, weather, culture: 0 of 5. Not representable from this sample; those traders are all Limited.

Three most instructive visible handles:
1. szg.szg (monthly #42, volume #49): 100% KXBTC15M across only 3 tickers, median fill 100 contracts, 62% taker / 38% maker over a one-hour window. Looks like an automated rolling-window trader/market-maker on the BTC 15-minute series, the closest analogue to your bots, and visible fill-by-fill.
2. BWmarkets (all-time #60, ~$963k all-time pnl): only KXNCAAFTOTAL (66%) and KXNCAAFSPREAD (34%), 4 tickers, median 111 contracts, max 64.9k, 70% taker. A concentrated college-football totals/spread specialist whose fills show size and timing for each game.
3. BSharp (monthly #49, weekly sports #4): 10 series but 52% KXCS2GAME (CS2 esports), then NCAAM/NBA spreads and totals and NFL fantasy points; 100% taker, median 99 contracts. A multi-sport spread shopper that takes liquidity.
(far.bike and Lajollawinner trade only cross-category combos with very large size, median 12k and 4k contracts, max 314k and 286k, all taker, so they are combos takers rather than strategy templates.)

Maker/taker: the fill record gives maker_nickname vs taker_nickname, so role is known per fill; 4 of 5 visible handles are 100% or majority taker.
