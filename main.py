"""Launch-day Twitch CCU forecaster for gta6-twitch-ccu-forecast-01.

Idea in one paragraph: the only hard fact at forecast time is hour 1's print,
so every later hour is expressed as a ratio to that print. The case file
carries dozens of past big launches with their full release days, and those
days trace a fairly stable shape once they are aligned on the UTC release
day: a soft dip in the small hours, a long climb through the day, a peak in
the late evening UTC (US prime time) and a gentle fall into hour 24. This
program learns that shape at run time from the comparables, hour by hour,
and reports honest quantiles of the ratio distribution instead of a single
guessed curve.

Which launches teach the shape matters more than the exact estimator:

* Only real launches are used. One comparable is the target category's own
  pre-release history (its "unlock" note says it is not a launch); it is
  dropped.
* A launch only teaches if its first UTC hour was already a launch hour.
  Launches are kept when hour 1 reached at least 15% of that day's peak.
  That keeps the rolling-midnight console launches (the shape this contest's
  cases all follow) and drops simultaneous global unlocks whose hour 1 is
  still pre-release noise with ratios in the hundreds.
* Launches are weighted by how close their hour-1 audience is to this
  case's hour-1 print, on a log scale: big launches bend differently from
  small ones, and the kernel lets the near neighbours speak loudest.

For each hour the five reported quantiles blend the weighted empirical
quantiles of the log ratios with a normal curve fitted to the weighted
interquartile range, the median is shrunk a few percent toward persistence,
and the whole set of curves is lightly smoothed across neighbouring hours.
Everything degrades gracefully: a thin comparable pool relaxes the maturity
cut step by step, and if nothing usable remains the forecast falls back to
persistence bands around hour 1. The program never raises on a case: a
malformed case still gets a valid, if plain, forecast file.

Run contract (from the kit): read every *.in in $MAKEFUN_INPUT_DIR (else
/input), write <id>.out to $MAKEFUN_OUTPUT_DIR (else /output), standard
library only, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_OF_TAU = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
HOURS = tuple(range(2, 25))

# A comparable teaches the shape only if its hour 1 was already mature:
# at least this share of its own release-day peak.
MATURITY_CUTS = (0.15, 0.08, 0.03, 0.0)
MIN_POOL = 5
MIN_ANCHOR = 100.0
# Log-scale bandwidth of the size-similarity kernel.
KERNEL_SIGMA = 1.0
# The median log ratio is pulled this far toward 0 (persistence).
CENTRE_SHRINK = 0.95
# Blend weight on the empirical quantile (the rest is the IQR normal fit);
# tails lean harder on the normal fit because order statistics are noisy.
EMPIRICAL_WEIGHT_INNER = 0.75
EMPIRICAL_WEIGHT_TAIL = 0.5
SD_FLOOR = 0.08
VALUE_CAP = 1_000_000_000.0


def parse_instant(text):
    """Parse an ISO-8601 UTC timestamp; return None when unreadable."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def round_cents(value):
    """Round to two decimals, half up, keeping whole numbers whole."""
    scaled = math.floor(value * 100.0 + 0.5) / 100.0
    return int(scaled) if float(scaled).is_integer() else scaled


def anchor_of(case):
    """Hour 1's print for the case, with defensive fallbacks; at least 1."""
    for row in case.get("known") or []:
        if row.get("hour") == 1 and row.get("viewers") is not None:
            try:
                return max(1.0, float(row["viewers"]))
            except (TypeError, ValueError):
                pass
    hourly = case.get("history_hourly") or []
    if hourly:
        last = hourly[-1].get("viewers")
        if last is not None:
            try:
                return max(1.0, float(last))
            except (TypeError, ValueError):
                pass
    return 1.0


def release_day_curve(comparable):
    """Map hour 1..24 -> viewers for one comparable's release day.

    Returns None when the comparable is not a launch, has no readable
    day_start, or lacks a usable hour 1.
    """
    note = str(comparable.get("unlock") or "")
    if note.startswith("Not a launch"):
        return None
    start = parse_instant(comparable.get("day_start"))
    if start is None:
        return None
    curve = {}
    for row in comparable.get("hourly") or []:
        when = parse_instant(row.get("from"))
        if when is None:
            continue
        offset = (when - start).total_seconds()
        if offset < 0 or offset >= 24 * 3600 or offset % 3600 != 0:
            continue
        viewers = row.get("viewers")
        if viewers is None:
            continue
        try:
            curve[int(offset // 3600) + 1] = float(viewers)
        except (TypeError, ValueError):
            continue
    if curve.get(1, 0.0) < MIN_ANCHOR:
        return None
    return curve


def weighted_quantile(sorted_pairs, tau):
    """Quantile of (value, weight) pairs sorted by value, CDF-interpolated."""
    total = sum(w for _, w in sorted_pairs)
    if total <= 0:
        return sorted_pairs[len(sorted_pairs) // 2][0]
    goal = tau * total
    walked = 0.0
    prev_value, prev_walked = sorted_pairs[0][0], 0.0
    for value, weight in sorted_pairs:
        walked += weight
        if walked >= goal:
            if weight <= 0:
                return value
            share = (goal - prev_walked) / weight
            return prev_value + share * (value - prev_value)
        prev_value, prev_walked = value, walked
    return sorted_pairs[-1][0]


def similarity_weight(curve, anchor, sigma):
    """Gaussian kernel on the log distance between hour-1 audiences."""
    distance = math.log(curve[1] / anchor) / sigma
    return math.exp(-0.5 * distance * distance)


def pool_weights(pool, anchor):
    """Kernel weights for the pool, widened when neighbours are scarce.

    The plain kernel can leave a case whose scale sits far from every
    comparable with almost no effective data. When the effective sample
    size (sum w)^2 / sum(w^2) drops under six, the bandwidth widens so the
    estimate leans on the whole pool instead of one or two launches.
    """
    weights = [similarity_weight(curve, anchor, KERNEL_SIGMA) for curve in pool]
    total = sum(weights)
    squares = sum(w * w for w in weights)
    if squares > 0 and (total * total) / squares < 6.0:
        weights = [similarity_weight(curve, anchor, KERNEL_SIGMA * 1.75) for curve in pool]
    return weights


def pool_for(case, anchor):
    """Comparable release-day curves, maturity-filtered with relaxation."""
    curves = []
    for comparable in case.get("comparables") or []:
        curve = release_day_curve(comparable)
        if curve is not None:
            curves.append(curve)
    for cut in MATURITY_CUTS:
        chosen = [c for c in curves if c[1] >= cut * max(c.values())]
        if len(chosen) >= MIN_POOL:
            return chosen
    return curves


def ratio_bands(case, anchor):
    """Five log-ratio quantiles per forecast hour, learned from the pool.

    Returns {hour: [log q05, ..., log q95]} or None when the pool cannot
    support an estimate for most hours.
    """
    pool = pool_for(case, anchor)
    if len(pool) < MIN_POOL:
        return None
    weights = pool_weights(pool, anchor)
    raw = {}
    for hour in HOURS:
        pairs = []
        for curve, weight in zip(pool, weights):
            if hour in curve:
                ratio = math.log(max(1.0, curve[hour]) / curve[1])
                pairs.append((ratio, weight))
        if len(pairs) < MIN_POOL:
            continue
        pairs.sort(key=lambda pair: pair[0])
        centre = weighted_quantile(pairs, 0.5)
        low_q = weighted_quantile(pairs, 0.25)
        high_q = weighted_quantile(pairs, 0.75)
        spread = max((high_q - low_q) / (2 * 0.674490), SD_FLOOR)
        band = []
        for tau, z in zip(TAUS, Z_OF_TAU):
            empirical = weighted_quantile(pairs, tau)
            fitted = centre + z * spread
            weight = EMPIRICAL_WEIGHT_TAIL if tau in (0.05, 0.95) else EMPIRICAL_WEIGHT_INNER
            band.append(weight * empirical + (1.0 - weight) * fitted)
        # Shrink the centre toward persistence, keep the band's width.
        band = [centre * CENTRE_SHRINK + (value - centre) for value in band]
        band[2] = centre * CENTRE_SHRINK
        raw[hour] = band
    if len(raw) < 12:
        return None
    # Light smoothing across neighbouring hours, per quantile level.
    smooth = {}
    for hour in HOURS:
        if hour not in raw:
            continue
        acc = [0.0] * len(TAUS)
        total = 0.0
        for near, weight in ((hour - 1, 0.25), (hour, 0.5), (hour + 1, 0.25)):
            if near in raw:
                for idx in range(len(TAUS)):
                    acc[idx] += weight * raw[near][idx]
                total += weight
        smooth[hour] = [value / total for value in acc]
    return smooth


def persistence_bands(anchor):
    """Fallback: every hour centred on hour 1, spread widening with time."""
    rows = []
    for hour in HOURS:
        qs = [anchor * math.exp(z * 0.15 * math.sqrt(hour - 1)) for z in Z_OF_TAU]
        rows.append({"hour": hour, "q": tidy(qs)})
    return rows


def tidy(values):
    """Clamp into the allowed range, force non-decreasing, round."""
    out = []
    for value in values:
        value = min(max(value, 0.01), VALUE_CAP)
        if out and value < out[-1]:
            value = out[-1]
        out.append(round_cents(value))
    return out


def forecast_rows(case):
    """The 23 forecast rows for one case; never raises."""
    anchor = anchor_of(case)
    try:
        bands = ratio_bands(case, anchor)
    except Exception:
        bands = None
    if bands is None:
        return persistence_bands(anchor)
    rows = []
    fallback = {row["hour"]: row["q"] for row in persistence_bands(anchor)}
    for hour in HOURS:
        if hour in bands:
            qs = [anchor * math.exp(value) for value in bands[hour]]
            rows.append({"hour": hour, "q": tidy(qs)})
        else:
            rows.append({"hour": hour, "q": fallback[hour]})
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            rows = forecast_rows(case)
        except Exception:
            rows = persistence_bands(1.0)
        payload = json.dumps({"hours": rows}, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
