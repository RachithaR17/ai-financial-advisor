import pandas as pd

CATEGORY_COLS = ['Rent', 'Groceries', 'Transport', 'Eating_Out', 'Entertainment',
                  'Utilities', 'Healthcare', 'Education', 'Miscellaneous']

def build_cohort_stats(df, category_cols=CATEGORY_COLS):
    """Precompute mean/std spending per City_Tier + Income_Bracket cohort."""
    df = df.copy()
    df['Income_Bracket'] = pd.qcut(df['Income'], q=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])
    cohort_stats = df.groupby(['City_Tier', 'Income_Bracket'])[category_cols].agg(['mean', 'std'])
    return df, cohort_stats

def benchmark_user(user_row, cohort_stats, category_cols=CATEGORY_COLS):
    """Compare one user's spending to their cohort. Returns dict of per-category stats."""
    cohort = cohort_stats.loc[(user_row['City_Tier'], user_row['Income_Bracket'])]
    results = {}
    for cat in category_cols:
        user_val = user_row[cat]
        cohort_mean = cohort[(cat, 'mean')]
        cohort_std = cohort[(cat, 'std')]
        z = (user_val - cohort_mean) / cohort_std if cohort_std > 0 else 0
        pct_diff = ((user_val - cohort_mean) / cohort_mean) * 100 if cohort_mean > 0 else 0
        results[cat] = {
            'user_spend': round(user_val, 2),
            'cohort_avg': round(cohort_mean, 2),
            'pct_diff_from_avg': round(pct_diff, 1),
            'z_score': round(z, 2)
        }
    return results
    