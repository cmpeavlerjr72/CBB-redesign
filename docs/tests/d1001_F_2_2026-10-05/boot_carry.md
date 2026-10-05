Paired game bootstrap (Decision 12), ref `v3full_COMB9GCTKD_s200_o0`, floor draws `d1001D_S2f1_s200_o1000`, `d1001D_S2f2_s200_o2000`, `d1001D_S2f3_s200_o3000`, `d1001D_S2f4_s200_o4000`; 2000 game resamples; floor = max(2 x draw SD, bootstrap 95% half-width); BEYOND = |move| > floor.


### fr1_FRc_box_v1_full_s200_o0_ev4_sv2 (`fr1_FRc_box_v1_full_s200_o0_ev4_sv2`) vs ref, 5705 common games

| gate | line | ref | arm | move | draw SD | 2 x draw SD | boot 95% CI of move | floor | move / floor | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | possessions/game mean (sim) | 68.8231 | 68.8160 | -0.0071 | 0.0041 | 0.0081 | [-0.0127, -0.0013] | 0.0081 | -0.88 | inside |
| G1 | possessions within-game SD (sim, mean over games) | 4.8115 | 4.7984 | -0.0131 | 0.0038 | 0.0075 | [-0.0208, -0.0050] | 0.0079 | -1.66 | BEYOND |
| G5 | margin SD ratio | 1.0419 | 1.0479 | +0.0059 | 0.0011 | 0.0021 | [+0.0007, +0.0109] | 0.0051 | +1.16 | BEYOND |
| G5 |   margin component: mean within-game sim SD | 12.1814 | 12.1759 | -0.0055 | 0.0031 | 0.0063 | [-0.0283, +0.0175] | 0.0229 | -0.24 | inside |
| G5 |   margin component: SD(actual - sim mean) | 11.6910 | 11.6198 | -0.0712 | 0.0107 | 0.0215 | [-0.1223, -0.0137] | 0.0543 | -1.31 | BEYOND |
| G5 | total SD ratio | 0.9254 | 0.9167 | -0.0087 | 0.0006 | 0.0011 | [-0.0110, -0.0065] | 0.0022 | -3.88 | BEYOND |
| G5 |   total component: mean within-game sim SD | 15.8366 | 15.6875 | -0.1491 | 0.0094 | 0.0187 | [-0.1693, -0.1290] | 0.0201 | -7.41 | BEYOND |
| G5 |   total component: SD(actual - sim mean) | 17.1131 | 17.1131 | -0.0001 | 0.0064 | 0.0129 | [-0.0362, +0.0349] | 0.0356 | -0.00 | inside |
| G5 |   accuracy: corr(actual total, sim total mean) | 0.3760 | 0.3758 | -0.0002 | 0.0008 | 0.0017 | [-0.0047, +0.0044] | 0.0046 | -0.04 | inside |
| G5 |   accuracy: corr(actual margin, sim margin mean) | 0.6027 | 0.6092 | +0.0065 | 0.0009 | 0.0018 | [+0.0016, +0.0109] | 0.0047 | +1.38 | BEYOND |
| G5 | home/away score correlation (sim) | 0.1260 | 0.1117 | -0.0143 | 0.0003 | 0.0005 | [-0.0174, -0.0114] | 0.0030 | -4.80 | BEYOND |
| G7 | OT rate (sim) | 0.0305 | 0.0299 | -0.0005 | 0.0001 | 0.0002 | [-0.0010, -0.0001] | 0.0004 | -1.19 | BEYOND |
| G9 | margin bias (sim - actual) | -0.2610 | -0.1878 | +0.0732 | 0.0062 | 0.0125 | [+0.0226, +0.1248] | 0.0511 | +1.43 | BEYOND |
| G9 | total bias (sim - actual) | -0.3372 | 0.4366 | +0.7738 | 0.0181 | 0.0362 | [+0.7381, +0.8103] | 0.0362 | +21.37 | BEYOND |
| G9 | calibration slope | 0.9479 | 0.9466 | -0.0014 | 0.0021 | 0.0043 | [-0.0106, +0.0076] | 0.0091 | -0.15 | inside |
| G4 | TOV% pooled (ratio of sums) | 0.1751 | 0.1724 | -0.0027 | 0.0000 | 0.0000 | [-0.0028, -0.0026] | 0.0001 | -20.59 | BEYOND |
| G4 | TOV count per team-game | 11.9367 | 11.7199 | -0.2168 | 0.0007 | 0.0015 | [-0.2260, -0.2076] | 0.0092 | -23.60 | BEYOND |
| G4 | OREB% pooled | 0.2887 | 0.2873 | -0.0014 | 0.0001 | 0.0001 | [-0.0016, -0.0012] | 0.0002 | -7.48 | BEYOND |
| G4 | eFG% pooled | 0.5080 | 0.5074 | -0.0006 | 0.0001 | 0.0002 | [-0.0008, -0.0005] | 0.0002 | -3.63 | BEYOND |
| G4 | FTA/FGA pooled | 0.3271 | 0.3664 | +0.0393 | 0.0001 | 0.0001 | [+0.0389, +0.0397] | 0.0004 | +100.66 | BEYOND |

### fr1_FRb_box_v1_full_s200_o0_ev4_sv2 (`fr1_FRb_box_v1_full_s200_o0_ev4_sv2`) vs ref, 5705 common games

| gate | line | ref | arm | move | draw SD | 2 x draw SD | boot 95% CI of move | floor | move / floor | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | possessions/game mean (sim) | 68.8231 | 68.8197 | -0.0033 | 0.0041 | 0.0081 | [-0.0093, +0.0024] | 0.0081 | -0.41 | inside |
| G1 | possessions within-game SD (sim, mean over games) | 4.8115 | 4.8030 | -0.0085 | 0.0038 | 0.0075 | [-0.0164, -0.0010] | 0.0077 | -1.11 | BEYOND |
| G5 | margin SD ratio | 1.0419 | 1.0468 | +0.0049 | 0.0011 | 0.0021 | [-0.0004, +0.0101] | 0.0053 | +0.92 | inside |
| G5 |   margin component: mean within-game sim SD | 12.1814 | 12.1760 | -0.0054 | 0.0031 | 0.0063 | [-0.0280, +0.0170] | 0.0225 | -0.24 | inside |
| G5 |   margin component: SD(actual - sim mean) | 11.6910 | 11.6316 | -0.0594 | 0.0107 | 0.0215 | [-0.1145, -0.0035] | 0.0555 | -1.07 | BEYOND |
| G5 | total SD ratio | 0.9254 | 0.9152 | -0.0102 | 0.0006 | 0.0011 | [-0.0126, -0.0079] | 0.0023 | -4.40 | BEYOND |
| G5 |   total component: mean within-game sim SD | 15.8366 | 15.6677 | -0.1689 | 0.0094 | 0.0187 | [-0.1893, -0.1489] | 0.0202 | -8.38 | BEYOND |
| G5 |   total component: SD(actual - sim mean) | 17.1131 | 17.1198 | +0.0067 | 0.0064 | 0.0129 | [-0.0300, +0.0448] | 0.0374 | +0.18 | inside |
| G5 |   accuracy: corr(actual total, sim total mean) | 0.3760 | 0.3749 | -0.0010 | 0.0008 | 0.0017 | [-0.0059, +0.0037] | 0.0048 | -0.21 | inside |
| G5 |   accuracy: corr(actual margin, sim margin mean) | 0.6027 | 0.6084 | +0.0057 | 0.0009 | 0.0018 | [+0.0010, +0.0103] | 0.0046 | +1.22 | BEYOND |
| G5 | home/away score correlation (sim) | 0.1260 | 0.1091 | -0.0169 | 0.0003 | 0.0005 | [-0.0201, -0.0139] | 0.0031 | -5.52 | BEYOND |
| G7 | OT rate (sim) | 0.0305 | 0.0302 | -0.0002 | 0.0001 | 0.0002 | [-0.0007, +0.0002] | 0.0004 | -0.56 | inside |
| G9 | margin bias (sim - actual) | -0.2610 | -0.1438 | +0.1171 | 0.0062 | 0.0125 | [+0.0657, +0.1661] | 0.0502 | +2.33 | BEYOND |
| G9 | total bias (sim - actual) | -0.3372 | 0.4252 | +0.7624 | 0.0181 | 0.0362 | [+0.7268, +0.7994] | 0.0363 | +21.01 | BEYOND |
| G9 | calibration slope | 0.9479 | 0.9398 | -0.0081 | 0.0021 | 0.0043 | [-0.0174, +0.0008] | 0.0091 | -0.90 | inside |
| G4 | TOV% pooled (ratio of sums) | 0.1751 | 0.1726 | -0.0025 | 0.0000 | 0.0000 | [-0.0026, -0.0023] | 0.0001 | -19.11 | BEYOND |
| G4 | TOV count per team-game | 11.9367 | 11.7356 | -0.2011 | 0.0007 | 0.0015 | [-0.2104, -0.1920] | 0.0092 | -21.90 | BEYOND |
| G4 | OREB% pooled | 0.2887 | 0.2874 | -0.0014 | 0.0001 | 0.0001 | [-0.0016, -0.0012] | 0.0002 | -7.19 | BEYOND |
| G4 | eFG% pooled | 0.5080 | 0.5076 | -0.0005 | 0.0001 | 0.0002 | [-0.0006, -0.0003] | 0.0002 | -2.74 | BEYOND |
| G4 | FTA/FGA pooled | 0.3271 | 0.3657 | +0.0387 | 0.0001 | 0.0001 | [+0.0383, +0.0391] | 0.0004 | +99.36 | BEYOND |

### fr1_FTc_box_v1_full_s200_o0_ev4_sv2 (`fr1_FTc_box_v1_full_s200_o0_ev4_sv2`) vs ref, 5705 common games

| gate | line | ref | arm | move | draw SD | 2 x draw SD | boot 95% CI of move | floor | move / floor | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | possessions/game mean (sim) | 68.8231 | 68.8153 | -0.0077 | 0.0041 | 0.0081 | [-0.0151, -0.0003] | 0.0081 | -0.95 | inside |
| G1 | possessions within-game SD (sim, mean over games) | 4.8115 | 4.8016 | -0.0099 | 0.0038 | 0.0075 | [-0.0175, -0.0020] | 0.0077 | -1.28 | BEYOND |
| G5 | margin SD ratio | 1.0419 | 1.0464 | +0.0044 | 0.0011 | 0.0021 | [-0.0019, +0.0105] | 0.0062 | +0.71 | inside |
| G5 |   margin component: mean within-game sim SD | 12.1814 | 12.1701 | -0.0113 | 0.0031 | 0.0063 | [-0.0353, +0.0111] | 0.0232 | -0.49 | inside |
| G5 |   margin component: SD(actual - sim mean) | 11.6910 | 11.6307 | -0.0603 | 0.0107 | 0.0215 | [-0.1255, +0.0070] | 0.0663 | -0.91 | inside |
| G5 | total SD ratio | 0.9254 | 0.9166 | -0.0088 | 0.0006 | 0.0011 | [-0.0118, -0.0057] | 0.0031 | -2.87 | BEYOND |
| G5 |   total component: mean within-game sim SD | 15.8366 | 15.6992 | -0.1374 | 0.0094 | 0.0187 | [-0.1576, -0.1161] | 0.0208 | -6.62 | BEYOND |
| G5 |   total component: SD(actual - sim mean) | 17.1131 | 17.1276 | +0.0144 | 0.0064 | 0.0129 | [-0.0377, +0.0640] | 0.0509 | +0.28 | inside |
| G5 |   accuracy: corr(actual total, sim total mean) | 0.3760 | 0.3740 | -0.0019 | 0.0008 | 0.0017 | [-0.0083, +0.0048] | 0.0066 | -0.29 | inside |
| G5 |   accuracy: corr(actual margin, sim margin mean) | 0.6027 | 0.6080 | +0.0053 | 0.0009 | 0.0018 | [-0.0006, +0.0109] | 0.0057 | +0.92 | inside |
| G5 | home/away score correlation (sim) | 0.1260 | 0.1173 | -0.0088 | 0.0003 | 0.0005 | [-0.0123, -0.0053] | 0.0035 | -2.53 | BEYOND |
| G7 | OT rate (sim) | 0.0305 | 0.0301 | -0.0004 | 0.0001 | 0.0002 | [-0.0008, +0.0000] | 0.0004 | -0.94 | inside |
| G9 | margin bias (sim - actual) | -0.2610 | -0.2398 | +0.0212 | 0.0062 | 0.0125 | [-0.0434, +0.0853] | 0.0644 | +0.33 | inside |
| G9 | total bias (sim - actual) | -0.3372 | 0.4136 | +0.7508 | 0.0181 | 0.0362 | [+0.6997, +0.8035] | 0.0519 | +14.46 | BEYOND |
| G9 | calibration slope | 0.9479 | 0.9519 | +0.0040 | 0.0021 | 0.0043 | [-0.0078, +0.0151] | 0.0115 | +0.35 | inside |
| G4 | TOV% pooled (ratio of sums) | 0.1751 | 0.1731 | -0.0020 | 0.0000 | 0.0000 | [-0.0022, -0.0017] | 0.0002 | -8.04 | BEYOND |
| G4 | TOV count per team-game | 11.9367 | 11.7699 | -0.1668 | 0.0007 | 0.0015 | [-0.1839, -0.1500] | 0.0170 | -9.83 | BEYOND |
| G4 | OREB% pooled | 0.2887 | 0.2874 | -0.0013 | 0.0001 | 0.0001 | [-0.0017, -0.0009] | 0.0004 | -3.13 | BEYOND |
| G4 | eFG% pooled | 0.5080 | 0.5080 | -0.0001 | 0.0001 | 0.0002 | [-0.0003, +0.0001] | 0.0002 | -0.35 | inside |
| G4 | FTA/FGA pooled | 0.3271 | 0.3657 | +0.0386 | 0.0001 | 0.0001 | [+0.0380, +0.0393] | 0.0007 | +58.80 | BEYOND |

### fr1_FTb_box_v2_full_s200_o0_ev4_sv2 (`fr1_FTb_box_v2_full_s200_o0_ev4_sv2`) vs ref, 5705 common games

| gate | line | ref | arm | move | draw SD | 2 x draw SD | boot 95% CI of move | floor | move / floor | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| G1 | possessions/game mean (sim) | 68.8231 | 68.8156 | -0.0075 | 0.0041 | 0.0081 | [-0.0148, -0.0001] | 0.0081 | -0.92 | inside |
| G1 | possessions within-game SD (sim, mean over games) | 4.8115 | 4.8037 | -0.0078 | 0.0038 | 0.0075 | [-0.0154, -0.0003] | 0.0076 | -1.03 | BEYOND |
| G5 | margin SD ratio | 1.0419 | 1.0452 | +0.0033 | 0.0011 | 0.0021 | [-0.0028, +0.0094] | 0.0061 | +0.54 | inside |
| G5 |   margin component: mean within-game sim SD | 12.1814 | 12.1609 | -0.0205 | 0.0031 | 0.0063 | [-0.0434, +0.0025] | 0.0230 | -0.89 | inside |
| G5 |   margin component: SD(actual - sim mean) | 11.6910 | 11.6347 | -0.0563 | 0.0107 | 0.0215 | [-0.1237, +0.0084] | 0.0660 | -0.85 | inside |
| G5 | total SD ratio | 0.9254 | 0.9164 | -0.0090 | 0.0006 | 0.0011 | [-0.0119, -0.0059] | 0.0030 | -2.99 | BEYOND |
| G5 |   total component: mean within-game sim SD | 15.8366 | 15.6745 | -0.1621 | 0.0094 | 0.0187 | [-0.1812, -0.1416] | 0.0198 | -8.17 | BEYOND |
| G5 |   total component: SD(actual - sim mean) | 17.1131 | 17.1036 | -0.0095 | 0.0064 | 0.0129 | [-0.0616, +0.0402] | 0.0509 | -0.19 | inside |
| G5 |   accuracy: corr(actual total, sim total mean) | 0.3760 | 0.3772 | +0.0013 | 0.0008 | 0.0017 | [-0.0051, +0.0080] | 0.0065 | +0.19 | inside |
| G5 |   accuracy: corr(actual margin, sim margin mean) | 0.6027 | 0.6077 | +0.0050 | 0.0009 | 0.0018 | [-0.0005, +0.0107] | 0.0056 | +0.89 | inside |
| G5 | home/away score correlation (sim) | 0.1260 | 0.1170 | -0.0090 | 0.0003 | 0.0005 | [-0.0126, -0.0054] | 0.0036 | -2.49 | BEYOND |
| G7 | OT rate (sim) | 0.0305 | 0.0301 | -0.0003 | 0.0001 | 0.0002 | [-0.0008, +0.0001] | 0.0004 | -0.74 | inside |
| G9 | margin bias (sim - actual) | -0.2610 | -0.2149 | +0.0461 | 0.0062 | 0.0125 | [-0.0162, +0.1066] | 0.0614 | +0.75 | inside |
| G9 | total bias (sim - actual) | -0.3372 | 0.4034 | +0.7407 | 0.0181 | 0.0362 | [+0.6875, +0.7936] | 0.0531 | +13.96 | BEYOND |
| G9 | calibration slope | 0.9479 | 0.9503 | +0.0024 | 0.0021 | 0.0043 | [-0.0089, +0.0139] | 0.0114 | +0.21 | inside |
| G4 | TOV% pooled (ratio of sums) | 0.1751 | 0.1731 | -0.0019 | 0.0000 | 0.0000 | [-0.0022, -0.0017] | 0.0002 | -7.91 | BEYOND |
| G4 | TOV count per team-game | 11.9367 | 11.7729 | -0.1639 | 0.0007 | 0.0015 | [-0.1805, -0.1469] | 0.0168 | -9.76 | BEYOND |
| G4 | OREB% pooled | 0.2887 | 0.2874 | -0.0013 | 0.0001 | 0.0001 | [-0.0018, -0.0009] | 0.0004 | -3.24 | BEYOND |
| G4 | eFG% pooled | 0.5080 | 0.5080 | -0.0000 | 0.0001 | 0.0002 | [-0.0002, +0.0002] | 0.0002 | -0.11 | inside |
| G4 | FTA/FGA pooled | 0.3271 | 0.3650 | +0.0380 | 0.0001 | 0.0001 | [+0.0373, +0.0386] | 0.0007 | +58.22 | BEYOND |
