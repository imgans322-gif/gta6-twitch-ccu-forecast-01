# GTA VI launch-day Twitch CCU forecast

![Competition badge](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![Rank badge](https://make.fun/badge/gta6-twitch-ccu-forecast-01/imgans322-gif/gta6-twitch-ccu-forecast-01/rank.svg)
![Reward badge](https://make.fun/badge/gta6-twitch-ccu-forecast-01/imgans322-gif/gta6-twitch-ccu-forecast-01/reward.svg)

Entry for the make.fun competition **gta6-twitch-ccu-forecast-01**: forecast
the Twitch concurrent viewers of Grand Theft Auto VI in each of hours 2–24
of its release day (2026-11-19, UTC) as five quantiles per hour, given
everything recorded up to the end of hour 1. Scored by mean pinball loss on
the log scale. `main.py` (Python 3, standard library only) reads every
`*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes `<id>.out`
files to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Approach

Everything is anchored on hour 1's print: each later hour is forecast as a
ratio to that print, learned at run time from the past big launches shipped
inside the case file itself.

- **Comparable selection.** Only real launches teach the shape (the target
  category's own pre-release history is excluded), and only launches whose
  first UTC hour was already a genuine launch hour — hour 1 at least 15% of
  that day's peak. That keeps rolling-midnight console launches, the shape
  every graded case follows, and drops simultaneous global unlocks whose
  hour 1 is pre-release noise. If the pool runs thin, the cut relaxes step
  by step.
- **Size weighting.** Each comparable is weighted by a Gaussian kernel on
  the log distance between its hour-1 audience and this case's hour-1
  print, so launches of similar scale dominate; the kernel widens
  automatically when the effective sample size is small.
- **Quantiles.** Per hour, the five quantiles blend the weighted empirical
  quantiles of the log ratios with a normal curve fitted to the weighted
  interquartile range (tails lean more on the fit). The median is shrunk 5%
  toward persistence and the quantile curves are lightly smoothed across
  neighbouring hours. The learned shape dips in the small hours, climbs
  through the day and peaks around hours 20–21 UTC.
- **Robustness.** Missing or zero hour-1 prints, missing hours and empty
  comparable pools all degrade to persistence bands around the anchor; the
  program never crashes and always emits a spec-shaped forecast.

Validated by leave-one-out backtests over the comparable launches (the
shipped code path, mature-launch population): mean log pinball loss 0.211
versus 0.284 for the persistence baseline on the same cases, with 90%
interval coverage of 0.905. On the four practice cases the program runs in
well under a second in total and produces byte-identical output across
repeated runs.

## Layout

- `main.py` — the whole forecaster.
- `LICENSE` — MIT, from the competition kit's starter.
- `.makefun/owner.json` — entrant wallet for the claim file.
