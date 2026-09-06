import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "benchmarker"))
sys.path.append(os.path.join(os.path.dirname(__file__), "recommender"))

from benchmarker import build_cohort_stats, benchmark_user
from recommender import recommend_budget


def run_for_user(user_row, cohort_stats):
    print("=" * 60)
    print(f"USER PROFILE — Income: ₹{user_row['Income']:.2f}, "
          f"City Tier: {user_row['City_Tier']}, Age: {user_row['Age']}")
    print("=" * 60)

    print("\n--- PEER BENCHMARKING ---")
    bench = benchmark_user(user_row, cohort_stats)
    for cat, stats in bench.items():
        print(f"{cat}: ₹{stats['user_spend']} vs cohort avg ₹{stats['cohort_avg']} "
              f"({stats['pct_diff_from_avg']:+.1f}%)")

    print("\n--- BUDGET RECOMMENDATION ---")
    rec = recommend_budget(user_row)
    print(rec['message'])
    if rec['allocation']:
        for cat, amt in rec['allocation'].items():
            old = user_row[cat]
            change = amt - old
            if abs(change) > 0.01:
                print(f"{cat}: ₹{old:.2f} → ₹{amt:.2f}  ({change:+.2f})")


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "data.csv")
    df = pd.read_csv(data_path)
    df, cohort_stats = build_cohort_stats(df)

    typical_idx = 0
    slack = (df['Income'] - df[['Rent', 'Loan_Repayment', 'Insurance']].sum(axis=1)
              - df['Desired_Savings'] - df[['Groceries', 'Transport', 'Eating_Out',
              'Entertainment', 'Utilities', 'Healthcare', 'Education', 'Miscellaneous']].sum(axis=1))
    deficit_idx = slack.idxmin()

    run_for_user(df.iloc[typical_idx], cohort_stats)
    print("\n\n")
    run_for_user(df.iloc[deficit_idx], cohort_stats)