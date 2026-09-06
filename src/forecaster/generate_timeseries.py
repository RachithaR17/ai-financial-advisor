import numpy as np
import pandas as pd

CATEGORY_COLS = ['Rent', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment',
                  'Utilities', 'Healthcare', 'Education', 'Miscellaneous']

def generate_user_timeseries(user_row, n_months=12, category_cols=CATEGORY_COLS, seed=None):
    """
    Generate a plausible n_months history for one user, seeded from their
    real profile. Adds seasonal noise + a mild random trend per category.
    This is SYNTHETIC data, generated to compensate for the primary dataset
    having no time dimension — documented as such in the project report.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_months, freq='MS')
    data = {'month': months}
    for cat in category_cols:
        base = user_row[cat]
        # mild random monthly trend: total drift of -10% to +10% over the whole window
        trend_slope = rng.uniform(-0.10, 0.10) * base / n_months
        # month-to-month noise: ~8% of the base value, roughly realistic volatility
        noise = rng.normal(0, base * 0.08, size=n_months)
        trend = trend_slope * np.arange(n_months)
        series = base + trend + noise
        series = np.clip(series, a_min=0, a_max=None)  # spending can't go negative
        data[cat] = series

    return pd.DataFrame(data)


if __name__ == "__main__":
    import os
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "data.csv")
    df = pd.read_csv(data_path)

    sample_user = df.iloc[0]
    ts = generate_user_timeseries(sample_user, seed=42)
    print(ts[['month', 'Groceries', 'Transport', 'Eating_Out']])