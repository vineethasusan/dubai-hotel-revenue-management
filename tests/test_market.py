import numpy as np
import pandas as pd

from hotelrm.market import HOTELS, calendar


def test_events_and_ramadan_on_calendar():
    c = calendar("2025-01-01", "2025-12-31").set_index("date")
    assert "GITEX Global" in c.loc["2025-10-14", "event"]
    assert c.loc["2025-03-10", "ramadan"] == 1 and c.loc["2025-04-15", "ramadan"] == 0
    assert c.loc["2025-01-15", "season"] > c.loc["2025-07-15", "season"]


def test_sales_never_exceed_capacity_and_demand_falls_with_price(data):
    df, mkt = data
    assert (df["rooms_sold"] <= df["rooms"]).all()
    h = "Deira Midscale Inn"
    dates = pd.DatetimeIndex(["2025-09-10"] * 2)
    lo, hi = mkt.true_demand(h, dates, np.array([300.0, 450.0]))
    assert hi < lo
