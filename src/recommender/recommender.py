import numpy as np
from scipy.optimize import minimize

FIXED_COLS = ['Rent', 'Loan_Repayment', 'Insurance']
FLEXIBLE_COLS = ['Groceries', 'Transport', 'Eating_Out', 'Entertainment',
                  'Utilities', 'Healthcare', 'Education', 'Miscellaneous']

def recommend_budget(user_row, fixed_cols=FIXED_COLS, flexible_cols=FLEXIBLE_COLS, min_ratio=0.5):
    """Recommend a flexible-spending allocation that meets the user's savings goal."""
    income = user_row['Income']
    fixed_total = sum(user_row[c] for c in fixed_cols)
    savings_goal = user_row['Desired_Savings']
    current_flex = np.array([user_row[c] for c in flexible_cols])
    current_flex_total = current_flex.sum()

    available_for_flex = income - fixed_total - savings_goal

    if available_for_flex >= current_flex_total:
        return {
            'feasible': True,
            'message': "Current spending already meets the savings goal — no changes needed.",
            'allocation': dict(zip(flexible_cols, current_flex.round(2)))
        }

    lower_bounds = current_flex * min_ratio
    if available_for_flex < lower_bounds.sum():
        return {
            'feasible': False,
            'message': "Savings goal isn't reachable by adjusting flexible spending alone — fixed costs or the goal itself need review.",
            'allocation': None
        }

    def objective(x):
        return np.sum((x - current_flex) ** 2)

    constraints = [{'type': 'eq', 'fun': lambda x: x.sum() - available_for_flex}]
    bounds = [(lower_bounds[i], current_flex[i]) for i in range(len(flexible_cols))]

    result = minimize(objective, current_flex, bounds=bounds, constraints=constraints, method='SLSQP')

    return {
        'feasible': True,
        'message': f"Adjusted flexible spending to free up ₹{current_flex_total - available_for_flex:.2f}/month.",
        'allocation': dict(zip(flexible_cols, result.x.round(2)))
    }