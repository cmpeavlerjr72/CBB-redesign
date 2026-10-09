**What matches (powered cells).** The shape of how games unfold is right on the headline measures. Mean |margin| at every
4-minute mark is within 0.4 points of real (CIs all include 0), SD of |margin| within 0.5 at every mark, lead changes
4.66 vs 4.61 real, ties 3.10 vs 3.05, largest lead 17.5 vs 17.4, largest scoring run 10.5 vs 10.5, comeback rate after
trailing by 10+ at the half 7.8% vs 7.3% (137 real eligible games), second-half minus first-half points 8.16 vs 8.09,
and possessions per 4-minute segment match in 8 of 10 segments (the two exceptions are item 1 below). Lead changes fall with game lopsidedness in the sim as
they do in real play (section 4), but more weakly (see below).

**What does not (flags, n = 371 games x 50 seeds).**
1. The end of each period. The sim plays about 0.6 more possessions per game in minutes 16-20 and 1.0 more in minutes
   36-40 than the parsed real games (CIs exclude 0), and in games that are not close at 2:00 (|margin| > 5, 235 real
   games) it scores 10.7 points in the last two minutes against 9.1 real (diff -1.6, CI -2.2 to -1.0). Close games at
   2:00 match (9.9 vs 9.7). Matching it, the sim finishes games decided slightly more often (decided by 2:00: 82.6% vs
   78.7%; by 4:00: 77.7% vs 74.4%; both CIs just touch 0) and has fewer |margin| <= 5 games at 2:00 (31.9% vs 36.7%,
   CI -0.1 to 9.7 points). Caveat: real possessions come from pbp segmentation, which may not count a final
   sub-possession tail that the engine always resolves (the engine's censored last possession "consumes what is left
   and still resolves"), so part of the possession gap may be a definition difference; the points gap in not-close games
   is not.
2. Overtime. 3.7% of sim games go to OT against 7.8% in this window and 5.6% over the whole 2024-25 season (window n =
   371, so the window rate is noisy; against the season rate the sim is still about 2 points low). This is consistent
   with item 1: fewer games are tied or one possession from tied late.
3. Home margin level. Real home margin at the final whistle is +3.7 vs +2.5 sim in this window (diff 1.2, CI 0.15 to
   2.4); the gap opens between minutes 8 and 12 and then stays. One window and one site mix; not a stable estimate.
4. Totals responsiveness. Real total points rise from 144 (closest-spread quintile) to 148-149 in lopsided games and to
   152 for the strongest home-team quintile; the sim stays at 145-146 in every cell (home-rating Q5: 146.3 vs 152.0,
   CI 2.2 to 9.9, n = 74). Slope across |spread| quintiles real +0.57 per step, sim +0.04. Lead changes slope across the
   same quintiles: real -0.52, sim -0.29 per step (continuous, per point of spread: real -0.188, sim -0.107, difference CI
   -0.160 to +0.008): the sim has the right sign but is about 40% flatter than real; borderline.

**Underpowered cells** (labelled in the tables): comeback by home-rating quintile (22-32 eligible games per cell),
last-2-min points by quintile (23-32), and the 15+ and 10-14 half-margin cells for last-2-min points (7 and 16 games).
They are not evidence either way. Of roughly 125 cells above, 10 CIs exclude 0; with 95% intervals about 6
would be expected by chance, so the flags above are the ones that also form a coherent story (items 1 and 2) or repeat
across slices (item 4), not isolated hits (e.g. the |margin| at 4:00 cell for half-margin 5-9).
=====
Candidates for a post-freeze round (nothing was changed here; each needs its own pre-registered bake-off):
1. Period-end handling: the late-game round-4 "no shot at the horn" law (`ENGINE_LG_BUZZER`, default OFF) and a
   possession-count check of the censored last possession against pbp (diagnose first: count real possessions that
   end with the clock expiring without a shot; compare with the engine's `possessions_censored_by_period_end`).
2. Garbage-time scoring and clock burning by the leader in not-close endgames (sim +1.6 points in the last two
   minutes when |margin| > 5); this is the late-game family, trajectory-level target: last-2-min points and possessions
   by margin at 2:00.
3. Overtime frequency: P(tied at end of regulation) is low (3.7% vs 5.6% season base rate); decompose into the final-two-minutes
   margin distribution (item 1/2) before touching any OT model.
4. Home-margin level and its onset between minutes 8 and 12 of the first half (check the site/home term in the
   event and clock models with a larger window; this one window cannot separate it from sampling).
5. Total-points responsiveness to team strength and to lopsidedness (sim totals flat across rating and spread
   quintiles; real rise): a matchup-specificity check at the game-aggregate level for the team-rate draw.
