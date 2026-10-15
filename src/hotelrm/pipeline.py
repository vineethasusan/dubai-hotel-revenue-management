"""Train on Jul 2024 - Dec 2025, test on Jan - Jun 2026: forecast accuracy, learned elasticities,
and a counterfactual comparison of legacy rate-card pricing vs model-driven dynamic pricing."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import models
from .market import HOTELS, history

OUT = Path("outputs")
SPLIT = "2026-01-01"


def simulate(mkt, test: pd.DataFrame, price: np.ndarray, seed: int = 99) -> pd.DataFrame:
    """Rooms sold under a given price path, using the true demand process with identical noise for every policy."""
    out = []
    for h, g in test.groupby("hotel", sort=False):
        rng = np.random.default_rng(seed + list(HOTELS).index(h))
        p = price[test.index.get_indexer(g.index)]
        d = mkt.true_demand(h, pd.DatetimeIndex(g["date"]), p, rng)
        sold = np.minimum(np.round(d), g["rooms"].to_numpy())
        out.append(pd.DataFrame({"hotel": h, "date": g["date"].to_numpy(), "price": p, "sold": sold, "rooms": g["rooms"].to_numpy()}))
    r = pd.concat(out, ignore_index=True)
    r["revenue"] = r["price"] * r["sold"]
    return r


def kpis(r: pd.DataFrame) -> dict:
    rn = r["sold"].sum()
    avail = r["rooms"].sum()
    return {"revenue": float(r["revenue"].sum()), "occupancy": float(rn / avail), "adr": float(r["revenue"].sum() / rn),
            "revpar": float(r["revenue"].sum() / avail)}


def run() -> dict:
    df, mkt = history()
    train, test = df[df["date"] < SPLIT].copy(), df[df["date"] >= SPLIT].reset_index(drop=True)

    occ = models.fit_occupancy(train)
    f_model = models.predict_occupancy(occ, test)
    f_pick = models.pickup_baseline(train, test)
    f_ly = models.last_year_baseline(df, test)
    ok = ~np.isnan(f_ly)
    forecast = {"model_mape": models.mape(test["rooms_sold"], f_model),
                "pickup_mape": models.mape(test["rooms_sold"], f_pick),
                "last_year_mape": models.mape(test["rooms_sold"][ok], f_ly[ok])}

    pr = models.fit_price_response(train)
    elas = {h: {"learned": round(models.elasticity(pr, test[test["hotel"] == h]), 2), "true": HOTELS[h][2]} for h in HOTELS}

    legacy = simulate(mkt, test, test["price"].to_numpy())
    dyn_price = models.optimise_prices(pr, test)
    dynamic = simulate(mkt, test, dyn_price)
    naive = simulate(mkt, test, models.optimise_prices(models.NaiveResponse().fit(train), test))
    oracle_price = optimal_true_prices(mkt, test)
    oracle = simulate(mkt, test, oracle_price)

    per_hotel = []
    for h in HOTELS:
        a, b = kpis(legacy[legacy["hotel"] == h]), kpis(dynamic[dynamic["hotel"] == h])
        per_hotel.append({"hotel": h, "segment": HOTELS[h][3], "legacy_revpar": round(a["revpar"]), "dynamic_revpar": round(b["revpar"]),
                          "revpar_uplift": round(b["revpar"] / a["revpar"] - 1, 4),
                          "legacy_occ": round(a["occupancy"], 3), "dynamic_occ": round(b["occupancy"], 3),
                          "legacy_adr": round(a["adr"]), "dynamic_adr": round(b["adr"])})
    L, D, O = kpis(legacy), kpis(dynamic), kpis(oracle)
    m = {"test_period": f"{SPLIT} to {test['date'].max().date()}", "hotels": len(HOTELS), "room_nights_available": int(test["rooms"].sum()),
         "forecast": {k: round(v, 4) for k, v in forecast.items()}, "elasticity": elas,
         "legacy": {k: round(v, 3) for k, v in L.items()}, "dynamic": {k: round(v, 3) for k, v in D.items()},
         "revenue_uplift": round(D["revenue"] / L["revenue"] - 1, 4), "revenue_gain_aed": round(D["revenue"] - L["revenue"]),
         "naive_model_revenue_uplift": round(kpis(naive)["revenue"] / L["revenue"] - 1, 4),
         "share_of_max_possible_gain": round((D["revenue"] - L["revenue"]) / (O["revenue"] - L["revenue"]), 4),
         "per_hotel": per_hotel}
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2))
    pd.DataFrame(per_hotel).to_csv(OUT / "hotel_results.csv", index=False)
    test.assign(dynamic_price=dyn_price, forecast_rooms=np.round(f_model)).to_csv(OUT / "pricing_decisions.csv.gz", index=False)
    (OUT / "report.md").write_text(report(m))
    return m


def optimal_true_prices(mkt, test: pd.DataFrame, grid=np.linspace(0.6, 2.2, 33)) -> np.ndarray:
    """Upper bound: the best grid price if the true demand curve were known (no noise)."""
    best_p, best_r = np.zeros(len(test)), np.full(len(test), -1.0)
    for h, g in test.groupby("hotel", sort=False):
        idx = test.index.get_indexer(g.index)
        ref, cap = HOTELS[h][1], g["rooms"].to_numpy()
        for x in grid:
            p = np.full(len(g), np.round(ref * x, -1))
            r = p * np.minimum(cap, mkt.true_demand(h, pd.DatetimeIndex(g["date"]), p))
            better = r > best_r[idx]
            best_p[idx[better]], best_r[idx[better]] = p[better], r[better]
    return best_p


def report(m: dict) -> str:
    f, L, D = m["forecast"], m["legacy"], m["dynamic"]
    lines = ["# Dubai Hotel Revenue Management: results", "", f"Test period {m['test_period']}, {m['hotels']} hotels, "
             f"{m['room_nights_available']:,} room-nights.", "", "## Occupancy forecast, 14 days before arrival", "",
             "| Method | MAPE |", "|---|---|", f"| Gradient boosting (calendar + events + pickup) | {f['model_mape']:.1%} |",
             f"| Additive pickup baseline | {f['pickup_mape']:.1%} |", f"| Same weekday last year | {f['last_year_mape']:.1%} |", "",
             "## Pricing: legacy rate card vs dynamic pricing (same true demand)", "", "| | Legacy | Dynamic |", "|---|---|---|",
             f"| Revenue (AED) | {L['revenue']:,.0f} | {D['revenue']:,.0f} |", f"| Occupancy | {L['occupancy']:.1%} | {D['occupancy']:.1%} |",
             f"| ADR (AED) | {L['adr']:,.0f} | {D['adr']:,.0f} |", f"| RevPAR (AED) | {L['revpar']:,.0f} | {D['revpar']:,.0f} |", "",
             f"Revenue uplift: **{m['revenue_uplift']:.1%}** (AED {m['revenue_gain_aed']:,.0f} over six months), "
             f"capturing {m['share_of_max_possible_gain']:.0%} of the gain available with perfect knowledge of demand.", "",
             f"For comparison, a naive model (price as just another feature, sold-out days dropped) would have changed revenue by "
             f"**{m['naive_model_revenue_uplift']:.1%}**: it under-estimates how price-sensitive guests are and over-prices.", "",
             "## By hotel", "", "| Hotel | Segment | RevPAR legacy | RevPAR dynamic | Uplift | Occ legacy | Occ dynamic |", "|---|---|---|---|---|---|---|"]
    for r in m["per_hotel"]:
        lines.append(f"| {r['hotel']} | {r['segment']} | {r['legacy_revpar']} | {r['dynamic_revpar']} | {r['revpar_uplift']:.1%} | "
                     f"{r['legacy_occ']:.0%} | {r['dynamic_occ']:.0%} |")
    lines += ["", "## Price elasticity learned vs true", "", "| Hotel | Learned | True |", "|---|---|---|"]
    lines += [f"| {h} | {v['learned']} | {v['true']} |" for h, v in m["elasticity"].items()]
    return "\n".join(lines)


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: v for k, v in m.items() if k != "per_hotel"}, indent=2))
