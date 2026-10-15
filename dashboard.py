"""Revenue manager's view: forecast and recommended prices by hotel, with a what-if price simulator."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from hotelrm import models
from hotelrm.market import HOTELS, history

st.set_page_config(page_title="Dubai Hotel Revenue Management", layout="wide")


@st.cache_resource
def load():
    df, mkt = history()
    train = df[df["date"] < "2026-01-01"]
    return df, models.fit_price_response(train)


df, pr = load()
st.title("Dubai Hotel Revenue Management")
m = json.loads(Path("outputs/metrics.json").read_text()) if Path("outputs/metrics.json").exists() else None
if m:
    c = st.columns(4)
    c[0].metric("Revenue uplift vs rate card", f"{m['revenue_uplift']:.1%}")
    c[1].metric("RevPAR", f"AED {m['dynamic']['revpar']:,.0f}", f"{m['dynamic']['revpar'] - m['legacy']['revpar']:+,.0f}")
    c[2].metric("Occupancy", f"{m['dynamic']['occupancy']:.0%}", f"{(m['dynamic']['occupancy'] - m['legacy']['occupancy']) * 100:+.1f} pts")
    c[3].metric("Forecast error (14 days out)", f"{m['forecast']['model_mape']:.1%}")

hotel = st.selectbox("Hotel", list(HOTELS))
h = df[(df["hotel"] == hotel) & (df["date"] >= "2026-01-01")].reset_index(drop=True)
h["recommended"] = models.optimise_prices(pr, h)
st.subheader("Rate card vs recommended price")
st.line_chart(h.set_index("date")[["price", "recommended"]].rename(columns={"price": "rate card"}))
st.dataframe(h[h["event"] != ""][["date", "event", "price", "recommended"]].rename(columns={"price": "rate card"}), hide_index=True)

st.subheader("What-if: price for one night")
day = st.date_input("Stay date", h["date"].iloc[40].date(), min_value=h["date"].min().date(), max_value=h["date"].max().date())
row = h[h["date"] == pd.Timestamp(day)]
ref = HOTELS[hotel][1]
grid = np.round(np.linspace(0.6, 2.0, 29) * ref, -1)
dem = models.predict_demand(pr, pd.concat([row] * len(grid), ignore_index=True), grid)
cap = HOTELS[hotel][0]
sim = pd.DataFrame({"price": grid, "rooms sold": np.minimum(dem, cap), "revenue": grid * np.minimum(dem, cap)}).set_index("price")
st.line_chart(sim["revenue"])
st.caption(f"Learned price elasticity for {hotel}: {pr.beta[hotel]:.2f}. Capacity {cap} rooms.")
