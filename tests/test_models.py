import numpy as np

from hotelrm import models
from hotelrm.market import HOTELS


def test_double_ml_recovers_elasticity(data):
    df, _ = data
    pr = models.fit_price_response(df[df["date"] < "2026-01-01"])
    for h, (_, _, true_e, _) in HOTELS.items():
        assert abs(pr.beta[h] - true_e) < 0.3, (h, pr.beta[h], true_e)


def test_forecast_beats_baselines(data):
    df, _ = data
    train, test = df[df["date"] < "2026-01-01"], df[df["date"] >= "2026-01-01"].reset_index(drop=True)
    m = models.fit_occupancy(train)
    assert models.mape(test["rooms_sold"], models.predict_occupancy(m, test)) < models.mape(test["rooms_sold"], models.pickup_baseline(train, test))


def test_optimiser_respects_floor_and_capacity_logic(data):
    df, _ = data
    test = df[df["date"] >= "2026-01-01"].reset_index(drop=True)
    pr = models.fit_price_response(df[df["date"] < "2026-01-01"])
    p = models.optimise_prices(pr, test)
    ref = test["hotel"].map(lambda h: HOTELS[h][1]).to_numpy()
    assert (p >= np.round(ref * 0.6, -1) - 1).all()


def test_dynamic_pricing_beats_rate_card(data):
    from hotelrm.pipeline import kpis, simulate
    df, mkt = data
    train, test = df[df["date"] < "2026-01-01"], df[df["date"] >= "2026-01-01"].reset_index(drop=True)
    pr = models.fit_price_response(train)
    legacy = kpis(simulate(mkt, test, test["price"].to_numpy()))
    dynamic = kpis(simulate(mkt, test, models.optimise_prices(pr, test)))
    assert dynamic["revenue"] > legacy["revenue"]
