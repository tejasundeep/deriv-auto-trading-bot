from datetime import datetime
import pandas as pd
import dukascopy_python as dp
from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_EUR_USD

# Fetch historical data
df = dp.fetch(
    instrument=INSTRUMENT_FX_MAJORS_EUR_USD,
    interval=dp.INTERVAL_MIN_1,
    offer_side=dp.OFFER_SIDE_BID,
    start=datetime(2025, 1, 1),
    end=datetime(2026, 5, 19),
    debug=True
)

df.to_csv("EURUSD_M1_ALL.csv")
print(f"Successfully downloaded {len(df)} rows.")
