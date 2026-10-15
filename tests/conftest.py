import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def data():
    from hotelrm.market import history
    df, mkt = history(start="2024-07-01", end="2026-03-31", seed=3)
    return df, mkt
