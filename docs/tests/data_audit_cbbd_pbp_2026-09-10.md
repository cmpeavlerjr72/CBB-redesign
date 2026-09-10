# CBBD play-by-play data audit -- 2026-09-10

Source: `data/raw/cbbd/pbp/plays_{season}.parquet`, pulled via `scripts/pull_cbbd_pbp.py` from `https://api.collegebasketballdata.com/plays/date`.

Coverage is checked against games with `status == "final"` in `data/raw/cbbd/games_{season}.parquet` (scheduled/cancelled/postponed games are expected to have no plays).

## Season 2022

- Games covered: 5831 / 5966 final games (missing: 135)
  - Example missing gameIds: [18426, 18446, 18498, 18576, 18604, 18667, 18676, 18695, 18711, 18749]
- Rows: 2,310,520
- onFloor completeness (exactly 5 home + 5 away ids): 0.00%
- shootingPlay share: 46.95%
- Rows with null clock: 0
- shotInfo field coverage (share non-null):
  - shot_shooter_id: 46.93%
  - shot_shooter_name: 46.93%
  - shot_made: 46.95%
  - shot_range: 46.95%
  - shot_assisted: 46.95%
  - shot_assisted_by_id: 8.11%
  - shot_assisted_by_name: 8.11%
  - shot_location_x: 29.49%
  - shot_location_y: 29.49%
- playType counts:
  - JumpShot: 543,640
  - Defensive Rebound: 367,188
  - LayUpShot: 261,964
  - MadeFreeThrow: 248,142
  - PersonalFoul: 238,679
  - Lost Ball Turnover: 185,837
  - Offensive Rebound: 160,844
  - Steal: 95,622
  - OfficialTVTimeOut: 57,930
  - Block Shot: 46,832
  - DunkShot: 29,218
  - ShortTimeOut: 25,911
  - Dead Ball Rebound: 17,947
  - End Period: 13,298
  - End Game: 6,584
  - RegularTimeOut: 4,229
  - Jumpball: 2,508
  - Technical Foul: 2,345
  - TipShot: 1,776
  - Not Available: 26

## Season 2023

- Games covered: 6115 / 6222 final games (missing: 107)
  - Example missing gameIds: [12186, 12194, 12329, 12332, 12351, 12386, 12394, 12498, 12507, 12514]
- Rows: 2,422,018
- onFloor completeness (exactly 5 home + 5 away ids): 0.00%
- shootingPlay share: 47.04%
- Rows with null clock: 0
- shotInfo field coverage (share non-null):
  - shot_shooter_id: 47.02%
  - shot_shooter_name: 47.02%
  - shot_made: 47.04%
  - shot_range: 47.04%
  - shot_assisted: 47.04%
  - shot_assisted_by_id: 8.13%
  - shot_assisted_by_name: 8.13%
  - shot_location_x: 30.73%
  - shot_location_y: 30.73%
- playType counts:
  - JumpShot: 557,417
  - Defensive Rebound: 377,201
  - LayUpShot: 279,839
  - MadeFreeThrow: 269,784
  - PersonalFoul: 253,298
  - Lost Ball Turnover: 192,000
  - Offensive Rebound: 171,223
  - Steal: 98,682
  - OfficialTVTimeOut: 61,143
  - Block Shot: 48,072
  - ShortTimeOut: 31,869
  - DunkShot: 30,810
  - Dead Ball Rebound: 18,310
  - End Period: 14,126
  - End Game: 6,934
  - RegularTimeOut: 4,613
  - Technical Foul: 3,324
  - Jumpball: 1,907
  - TipShot: 1,387
  - Not Available: 79

## Season 2024

- Games covered: 6151 / 6243 final games (missing: 92)
  - Example missing gameIds: [5910, 6032, 6209, 6271, 6275, 6299, 6362, 6390, 6398, 6448]
- Rows: 2,456,754
- onFloor completeness (exactly 5 home + 5 away ids): 90.16%
- shootingPlay share: 47.73%
- Rows with null clock: 0
- shotInfo field coverage (share non-null):
  - shot_shooter_id: 47.65%
  - shot_shooter_name: 47.65%
  - shot_made: 47.73%
  - shot_range: 47.73%
  - shot_assisted: 47.73%
  - shot_assisted_by_id: 8.06%
  - shot_assisted_by_name: 8.06%
  - shot_location_x: 32.07%
  - shot_location_y: 32.07%
- playType counts:
  - JumpShot: 566,084
  - Defensive Rebound: 381,043
  - LayUpShot: 290,521
  - MadeFreeThrow: 284,312
  - PersonalFoul: 251,979
  - Lost Ball Turnover: 180,929
  - Offensive Rebound: 180,650
  - Steal: 99,784
  - OfficialTVTimeOut: 61,816
  - Block Shot: 50,967
  - ShortTimeOut: 32,169
  - DunkShot: 30,782
  - Dead Ball Rebound: 15,387
  - End Period: 14,153
  - End Game: 6,887
  - RegularTimeOut: 4,568
  - Technical Foul: 2,448
  - Jumpball: 1,400
  - TipShot: 875

## Season 2025

- Games covered: 6135 / 6292 final games (missing: 157)
  - Example missing gameIds: [12, 14, 17, 18, 20, 21, 38, 39, 41, 42]
- Rows: 2,657,828
- onFloor completeness (exactly 5 home + 5 away ids): 98.07%
- shootingPlay share: 43.42%
- Rows with null clock: 0
- shotInfo field coverage (share non-null):
  - shot_shooter_id: 43.38%
  - shot_shooter_name: 43.38%
  - shot_made: 43.42%
  - shot_range: 43.42%
  - shot_assisted: 43.42%
  - shot_assisted_by_id: 7.49%
  - shot_assisted_by_name: 7.49%
  - shot_location_x: 32.61%
  - shot_location_y: 32.61%
- playType counts:
  - JumpShot: 562,281
  - Defensive Rebound: 367,969
  - MadeFreeThrow: 281,470
  - LayUpShot: 269,378
  - PersonalFoul: 247,798
  - Substitution: 232,751
  - Lost Ball Turnover: 178,612
  - Offensive Rebound: 175,223
  - Steal: 101,613
  - OfficialTVTimeOut: 61,156
  - Block Shot: 49,382
  - ShortTimeOut: 31,859
  - DunkShot: 29,548
  - Dead Ball Rebound: 22,295
  - End Period: 14,494
  - TipShot: 11,230
  - End Game: 6,998
  - Jumpball: 5,765
  - RegularTimeOut: 5,764
  - Technical Foul: 2,242

## Season 2026

- Games covered: 6262 / 6299 final games (missing: 37)
  - Example missing gameIds: [209813, 209839, 210058, 210086, 210140, 210760, 212725, 212843, 212890, 212894]
- Rows: 3,537,141
- onFloor completeness (exactly 5 home + 5 away ids): 96.99%
- shootingPlay share: 33.98%
- Rows with null clock: 0
- shotInfo field coverage (share non-null):
  - shot_shooter_id: 33.94%
  - shot_shooter_name: 33.94%
  - shot_made: 33.98%
  - shot_range: 33.98%
  - shot_assisted: 33.98%
  - shot_assisted_by_id: 5.84%
  - shot_assisted_by_name: 5.84%
  - shot_location_x: 22.27%
  - shot_location_y: 22.27%
- playType counts:
  - Substitution: 1,009,011
  - JumpShot: 547,719
  - Defensive Rebound: 370,277
  - MadeFreeThrow: 307,122
  - LayUpShot: 289,142
  - PersonalFoul: 261,770
  - Lost Ball Turnover: 178,580
  - Offensive Rebound: 163,185
  - Steal: 102,161
  - OfficialTVTimeOut: 63,100
  - Block Shot: 50,599
  - Dead Ball Rebound: 47,134
  - DunkShot: 34,271
  - ShortTimeOut: 32,549
  - TipShot: 23,618
  - Jumpball: 22,785
  - End Period: 15,675
  - End Game: 7,594
  - RegularTimeOut: 5,824
  - Technical Foul: 2,772
  - Coach's Challenge (Stands): 1,375
  - Coach's Challenge (Overturned): 877
  - Shot: 1

## Overall

- Total rows across all seasons: 13,384,261
- Union playType vocabulary size (event dictionary): 24
  - JumpShot: 2,777,141
  - Defensive Rebound: 1,863,678
  - LayUpShot: 1,390,844
  - MadeFreeThrow: 1,390,830
  - PersonalFoul: 1,253,524
  - Substitution: 1,241,762
  - Lost Ball Turnover: 915,958
  - Offensive Rebound: 851,125
  - Steal: 497,862
  - OfficialTVTimeOut: 305,145
  - Block Shot: 245,852
  - DunkShot: 154,629
  - ShortTimeOut: 154,357
  - Dead Ball Rebound: 121,073
  - End Period: 71,746
  - TipShot: 38,886
  - End Game: 34,997
  - Jumpball: 34,365
  - RegularTimeOut: 24,998
  - Technical Foul: 13,131
  - Coach's Challenge (Stands): 1,375
  - Coach's Challenge (Overturned): 877
  - Not Available: 105
  - Shot: 1

- Total API calls used (this manifest's running total): 1180
- X-CallLimit-Remaining (final observed): 28749
