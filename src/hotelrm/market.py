"""Synthetic Dubai hotel market: true demand process, events calendar and booking history.

The demand process is known (that is what lets us score pricing policies counterfactually):
  rooms demanded on a stay date = base x season x day-of-week x events x (price / ref price) ^ elasticity x noise
Bookings arrive over the 90 days before the stay date following a lead-time curve, and are capped by capacity.
History is generated under the hotel's legacy pricing: a fixed seasonal rate card with small manual tweaks.
Event dates are approximate and illustrative.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

HOTELS = {
    # name: (rooms, reference ADR in AED, elasticity, segment)
    "Palm Beach Resort": (420, 1450, -1.10, "luxury_resort"),
    "Marina Luxury Hotel": (360, 1150, -1.25, "luxury_city"),
    "Downtown Business Hotel": (300, 820, -1.45, "business"),
    "DIFC Executive Suites": (180, 950, -1.30, "business"),
    "Deira Midscale Inn": (260, 380, -1.85, "midscale"),
    "Al Barsha City Hotel": (240, 420, -1.75, "midscale"),
}

# (name, start, end, demand multiplier, which segments it lifts)
EVENTS = [
    ("New Year's Eve", "12-28", "01-02", 1.55, {"luxury_resort", "luxury_city", "midscale", "business"}),
    ("Dubai Shopping Festival", "12-15", "01-29", 1.18, {"luxury_city", "midscale", "luxury_resort"}),
    ("Gulfood", "02-17", "02-21", 1.35, {"business", "midscale"}),
    ("Arabian Travel Market", "05-05", "05-08", 1.25, {"business", "luxury_city"}),
    ("GITEX Global", "10-13", "10-17", 1.45, {"business", "luxury_city", "midscale"}),
    ("Dubai Airshow", "11-17", "11-21", 1.30, {"business", "luxury_city"}),
    ("F1 weekend (Abu Dhabi overflow)", "12-05", "12-08", 1.20, {"luxury_resort", "luxury_city"}),
]
RAMADAN = {2024: ("2024-03-11", "2024-04-09"), 2025: ("2025-03-01", "2025-03-30"), 2026: ("2026-02-18", "2026-03-19")}
EID_AL_ADHA = {2024: "2024-06-16", 2025: "2025-06-06", 2026: "2026-05-27"}
LEAD_DAYS = 90


def calendar(start: str, end: str) -> pd.DataFrame:
    d = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    md = d["date"].dt.strftime("%m-%d")
    d["event"] = ""
    d["event_lift"] = 1.0
    for name, s, e, lift, _ in EVENTS:
        m = (md >= s) & (md <= e) if s <= e else (md >= s) | (md <= e)
        d.loc[m, "event"] = np.where(d.loc[m, "event"] == "", name, d.loc[m, "event"] + " + " + name)
        d.loc[m, "event_lift"] = np.maximum(d.loc[m, "event_lift"], lift)
    d["ramadan"] = 0
    for s, e in RAMADAN.values():
        d.loc[(d["date"] >= s) & (d["date"] <= e), "ramadan"] = 1
    d["eid"] = 0
    for y, s in RAMADAN.items():
        end_r = pd.Timestamp(s[1])
        d.loc[(d["date"] > end_r) & (d["date"] <= end_r + pd.Timedelta(days=4)), "eid"] = 1
    for s in EID_AL_ADHA.values():
        d.loc[(d["date"] >= s) & (d["date"] <= pd.Timestamp(s) + pd.Timedelta(days=3)), "eid"] = 1
    doy = d["date"].dt.dayofyear
    d["season"] = 1.0 + 0.32 * np.cos(2 * np.pi * (doy - 15) / 365.25)      # peak mid-January, trough mid-July
    d["dow"] = d["date"].dt.dayofweek
    return d


def segment_event_lift(row_event: str, seg: str) -> float:
    lift = 1.0
    for name, _, _, l, segs in EVENTS:
        if name in row_event and seg in segs:
            lift = max(lift, l)
    return lift


def dow_factor(seg: str, dow: np.ndarray) -> np.ndarray:
    if seg == "business":                       # strong Mon-Thu, weak Fri-Sat
        return np.select([dow <= 3, dow == 4, dow == 5], [1.12, 0.85, 0.78], 0.95)
    return np.select([dow == 4, dow == 5], [1.18, 1.22], 0.94)    # leisure peaks Fri-Sat


@dataclass
class Market:
    cal: pd.DataFrame
    seed: int = 7

    def true_demand(self, hotel: str, dates: pd.DatetimeIndex, price: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        rooms, ref, elast, seg = HOTELS[hotel]
        c = self.cal.set_index("date").loc[dates]
        ev = np.array([segment_event_lift(e, seg) for e in c["event"]])
        ram = np.where(c["ramadan"].to_numpy() == 1, 0.80 if seg != "business" else 0.88, 1.0)
        eid = np.where(c["eid"].to_numpy() == 1, 1.25 if seg != "business" else 0.9, 1.0)
        base = 0.86 * rooms * c["season"].to_numpy() * dow_factor(seg, c["dow"].to_numpy()) * ev * ram * eid
        d = base * (np.asarray(price) / ref) ** elast
        if rng is not None:
            d = d * rng.lognormal(0, 0.08, len(d))
        return d


def legacy_price(hotel: str, cal: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Rate card: high/low season rates, +20% on known big events, small random manual overrides."""
    _, ref, _, _ = HOTELS[hotel]
    month = cal["date"].dt.month.to_numpy()
    high = np.isin(month, [11, 12, 1, 2, 3])
    shoulder = np.isin(month, [4, 10])
    p = ref * np.select([high, shoulder], [1.15, 1.0], 0.75)
    p = p * np.where(cal["event_lift"].to_numpy() >= 1.3, 1.2, 1.0)
    p = p * rng.uniform(0.92, 1.08, len(p))
    test = rng.random(len(p)) < 0.15                     # occasional rate tests: price moved -25%..+30%
    p = np.where(test, p * rng.uniform(0.75, 1.30, len(p)), p)
    return np.round(p, -1)


def history(start: str = "2024-07-01", end: str = "2026-06-30", seed: int = 7) -> tuple[pd.DataFrame, Market]:
    cal = calendar(start, end)
    mkt = Market(cal, seed)
    rng = np.random.default_rng(seed)
    lead_curve = np.exp(-np.arange(LEAD_DAYS + 1) / 24.0)[::-1]           # most bookings in the last 3-4 weeks
    lead_curve /= lead_curve.sum()
    cum = np.cumsum(lead_curve)
    rows = []
    for h, (rooms, ref, elast, seg) in HOTELS.items():
        price = legacy_price(h, cal, rng)
        demand = mkt.true_demand(h, pd.DatetimeIndex(cal["date"]), price, rng)
        sold = np.minimum(np.round(demand), rooms)
        # on-the-books snapshots at 30/14/7 days before arrival (pickup signal used for forecasting)
        def otb(days):
            frac = cum[LEAD_DAYS - days - 1]
            return np.minimum(np.round(demand * frac * rng.uniform(0.9, 1.1, len(demand))), rooms)
        rows.append(pd.DataFrame({"hotel": h, "segment": seg, "rooms": rooms, "date": cal["date"], "price": price,
                                  "rooms_sold": sold, "unconstrained": np.round(demand), "otb_30": otb(30), "otb_14": otb(14),
                                  "otb_7": otb(7), "event": cal["event"], "ramadan": cal["ramadan"], "eid": cal["eid"]}))
    df = pd.concat(rows, ignore_index=True)
    df["revenue"] = df["price"] * df["rooms_sold"]
    return df, mkt
