import numpy as np
import pandas as pd
from generate_timeseries import generate_user_timeseries, CATEGORY_COLS


def forecast_user(user_row, category_cols=CATEGORY_COLS, n_months_history=12, horizons=(1, 2, 3), seed=None):
    """
    Generate a synthetic history for this user, fit a simple linear trend
    per category, and project forward `horizons` months (e.g. 1, 2, 3 -> 30/60/90 days).
    Baseline forecaster — deliberately simple (no LSTM yet), documented as v0.
    """
    ts = generate_user_timeseries(user_row, n_months=n_months_history, category_cols=category_cols, seed=seed)
    x = np.arange(n_months_history)

    forecasts = {}
    for cat in category_cols:
        y = ts[cat].values
        slope, intercept = np.polyfit(x, y, 1)  # simple linear fit

        cat_forecasts = {}
        for h in horizons:
            future_x = n_months_history - 1 + h
            predicted = slope * future_x + intercept
            predicted = max(predicted, 0)  # spending can't go negative
            cat_forecasts[f"+{h}mo"] = round(predicted, 2)
        forecasts[cat] = {
            'current_avg': round(y[-3:].mean(), 2),  # last 3 months average, as "current" baseline
            'trend_per_month': round(slope, 2),
            'forecast': cat_forecasts
        }
    return forecasts


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "data.csv")
    df = pd.read_csv(data_path)

    sample_user = df.iloc[0]
    result = forecast_user(sample_user, seed=42)

    for cat, info in result.items():
        trend_word = "rising" if info['trend_per_month'] > 0 else "falling"
        print(f"{cat}: current ~₹{info['current_avg']}/mo, {trend_word} by ₹{abs(info['trend_per_month'])}/mo")
        for horizon, val in info['forecast'].items():
            print(f"   {horizon}: ₹{val}")