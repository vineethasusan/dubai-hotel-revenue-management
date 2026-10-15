"""Forecasting and pricing models.

1. Occupancy forecast 14 days before arrival (operations: staffing, overbooking, distribution).
   Gradient boosting on calendar + events + on-the-books pickup, against two industry baselines:
   the additive pickup method and same-weekday-last-year.
2. Price-response model (pricing): demand as a function of price and context, learned only from days
   that did not sell out (sold-out days hide true demand), with a monotonic constraint so higher prices
   never predict more demand.
3. Dynamic pricing: for each stay date, pick the price on a grid that maximises
   expected revenue = price x min(capacity, demand(price)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .market import HOTELS, segment_event_lift

SEGMENTS = sorted({v[3] for v in HOTELS.values()})


def features(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"]
    ref = df["hotel"].map(lambda h: HOTELS[h][1])
    X = pd.DataFrame({
        "hotel": df["hotel"].astype("category").cat.set_categories(list(HOTELS)).cat.codes,
        "segment": df["segment"].map(SEGMENTS.index),
        "rooms": df["rooms"],
        "dow": d.dt.dayofweek, "month": d.dt.month,
        "doy_sin": np.sin(2 * np.pi * d.dt.dayofyear / 365.25), "doy_cos": np.cos(2 * np.pi * d.dt.dayofyear / 365.25),
        "event_lift": [segment_event_lift(e, s) for e, s in zip(df["event"], df["segment"])],
        "ramadan": df["ramadan"], "eid": df["eid"],
        "log_price_ratio": np.log(df["price"] / ref),
    }, index=df.index)
    return X


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    m = y > 0
    return float(np.mean(np.abs(yhat[m] - y[m]) / y[m]))


# ---------- 1. occupancy forecast ----------

def fit_occupancy(train: pd.DataFrame):
    X = features(train).assign(otb_14=train["otb_14"] / train["rooms"])
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
    m.fit(X, train["rooms_sold"] / train["rooms"])
    return m


def predict_occupancy(m, df: pd.DataFrame) -> np.ndarray:
    X = features(df).assign(otb_14=df["otb_14"] / df["rooms"])
    return np.clip(m.predict(X), 0, 1) * df["rooms"].to_numpy()


def pickup_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Additive pickup: on-the-books at 14 days + average pickup from 14 days to arrival (by hotel and weekday)."""
    t = train.assign(pick=train["rooms_sold"] - train["otb_14"], dow=train["date"].dt.dayofweek)
    avg = t.groupby(["hotel", "dow"])["pick"].mean()
    key = list(zip(test["hotel"], test["date"].dt.dayofweek))
    return np.minimum(test["otb_14"].to_numpy() + avg.reindex(key).to_numpy(), test["rooms"].to_numpy())


def last_year_baseline(full: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Same hotel, same weekday, 364 days earlier."""
    look = full.set_index(["hotel", "date"])["rooms_sold"]
    key = list(zip(test["hotel"], test["date"] - pd.Timedelta(days=364)))
    return look.reindex(key).to_numpy()


# ---------- 2. price response ----------

def unconstrain(d: pd.DataFrame) -> np.ndarray:
    open_ = (d["rooms_sold"] < d["rooms"]) & (d["otb_30"] > 0)
    ratio = (d[open_]["rooms_sold"] / d[open_]["otb_30"]).groupby(d[open_]["hotel"]).median()
    est = d["otb_30"] * d["hotel"].map(ratio)
    sold_out = d["rooms_sold"] >= d["rooms"]
    return np.where(sold_out, np.maximum(d["rooms_sold"], est), d["rooms_sold"]).astype(float)


class PriceResponse:
    """log(demand) = a(context) + beta_hotel x log(price / reference price).

    beta is estimated with double machine learning: gradient boosting predicts both log demand and
    log price from context (season, events, weekday); the price effect is then estimated on the
    residuals with cross-fitting. This separates "we charged more because demand was high" from
    "demand fell because we charged more", which a plain model confuses (it under-estimates elasticity).
    Sold-out days hide true demand, so it is "unconstrained" from the 30-day booking pace
    (rooms on the books 30 days out x the hotel's usual ratio of final to 30-day bookings),
    a standard revenue-management technique; dropping those days instead would bias elasticity toward zero.
    """

    def __init__(self, folds: int = 3):
        self.folds = folds

    def _ctx(self, df):
        return features(df).drop(columns=["log_price_ratio"])

    def fit(self, train: pd.DataFrame) -> "PriceResponse":
        d = train.reset_index(drop=True)
        demand = unconstrain(d)
        X, y, p = self._ctx(d), np.log1p(demand), features(d)["log_price_ratio"].to_numpy()
        ry, rp = np.zeros(len(d)), np.zeros(len(d))
        fold = np.arange(len(d)) % self.folds
        for k in range(self.folds):
            tr, te = fold != k, fold == k
            gy = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=k).fit(X[tr], y[tr])
            gp = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=k).fit(X[tr], p[tr])
            ry[te], rp[te] = y[te] - gy.predict(X[te]), p[te] - gp.predict(X[te])
        self.beta = {}
        for h in HOTELS:
            m = (d["hotel"] == h).to_numpy()
            self.beta[h] = float(np.sum(ry[m] * rp[m]) / np.sum(rp[m] ** 2))
        # context model on price-adjusted demand: a(X) = log demand - beta * log price ratio
        b = d["hotel"].map(self.beta).to_numpy()
        self.base = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(X, y - b * p)
        return self

    def predict(self, df: pd.DataFrame, price: np.ndarray) -> np.ndarray:
        d = df.assign(price=price)
        b = d["hotel"].map(self.beta).to_numpy()
        return np.expm1(self.base.predict(self._ctx(d)) + b * features(d)["log_price_ratio"].to_numpy())


class NaiveResponse:
    """The common shortcut, kept for comparison: one gradient-boosting model with price as a feature,
    trained on days that did not sell out. It confuses cause and effect and under-states elasticity."""

    def fit(self, train: pd.DataFrame) -> "NaiveResponse":
        d = train[train["rooms_sold"] < train["rooms"]]
        X = features(d)
        cst = [-1 if c == "log_price_ratio" else 0 for c in X.columns]
        self.m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, monotonic_cst=cst, random_state=0)
        self.m.fit(X, np.log1p(d["rooms_sold"]))
        return self

    def predict(self, df: pd.DataFrame, price: np.ndarray) -> np.ndarray:
        return np.expm1(self.m.predict(features(df.assign(price=price))))


def fit_price_response(train: pd.DataFrame) -> PriceResponse:
    return PriceResponse().fit(train)


def predict_demand(m, df: pd.DataFrame, price: np.ndarray) -> np.ndarray:
    return m.predict(df, price)


def elasticity(m: PriceResponse, df: pd.DataFrame) -> float:
    return m.beta[df["hotel"].iloc[0]]


# ---------- 3. dynamic pricing ----------

def optimise_prices(m, df: pd.DataFrame, grid=np.linspace(0.6, 2.2, 33), floor_ratio: float = 0.6) -> np.ndarray:
    """Revenue-maximising grid price per stay date, never below floor_ratio x reference rate (brand protection)."""
    ref = df["hotel"].map(lambda h: HOTELS[h][1]).to_numpy()
    cap = df["rooms"].to_numpy()
    best_p, best_r = np.zeros(len(df)), np.full(len(df), -1.0)
    for g in grid:
        if g < floor_ratio:
            continue
        p = np.round(ref * g, -1)
        r = p * np.minimum(cap, predict_demand(m, df, p))
        better = r > best_r
        best_p[better], best_r[better] = p[better], r[better]
    return best_p
