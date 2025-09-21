import pulp
import numpy as np
from collections import defaultdict
from itertools import product
import random
import pandas as pd

df = pd.read_csv('team_week_win_prob.csv').set_index('Team').fillna(0)
print(df)

# Parameters
threshold = 0.5   # only allow picks with prob >= threshold

teams = list(df.index)
weeks = list(df.columns)

# Create LP problem
prob_lp = pulp.LpProblem("Pick_Teams_Per_Week", pulp.LpMaximize)

# Decision variables: only create variable if prob >= threshold and not NaN
x = {}
for t in teams:
    for w in weeks:
        val = df.loc[t, w]
        if pd.notna(val) and val >= threshold:
            var_name = f"x_{t}_{w}"
            x[(t, w)] = pulp.LpVariable(var_name, cat='Binary')
# Constraints:
# 1) At most one team per week
for w in weeks:
    vars_this_week = [x[(t, w)] for t in teams if (t, w) in x]
    if vars_this_week:
        prob_lp += pulp.lpSum(vars_this_week) <= 1, f"AtMostOneTeam_Week_{w}"
    else:
        # no eligible team for this week (all probs < threshold or NaN)
        # nothing to add; week will remain unassigned
        pass

# 2) Each team can be selected at most once across all weeks
for t in teams:
    vars_this_team = [x[(t, w)] for w in weeks if (t, w) in x]
    if vars_this_team:
        prob_lp += pulp.lpSum(vars_this_team) <= 1, f"AtMostOneWeek_Team_{t}"

# Objective: maximize sum of selected probabilities
objective_terms = []
for (t, w), var in x.items():
    objective_terms.append(df.loc[t, w] * var)
prob_lp += pulp.lpSum(objective_terms)

# hard-code existing selections
prob_lp += x[('DEN', 'Week 1')] == 1 # picked broncos in week 1
prob_lp += x[('BAL', 'Week 2')] == 1 # picked ravens in week 2
prob_lp += x[('SEA', 'Week 3')] == 1 # picked seahawks in week 3

# Solve
solver = pulp.PULP_CBC_CMD(msg=False)  # change msg=True to get solver log
prob_lp.solve(solver)

# Collect solution
picks = []
total = 0.0
for (t, w), var in x.items():
    if pulp.value(var) == 1:
        picks.append((w, t, float(df.loc[t, w])))
        total += float(df.loc[t, w])

# Sort picks by week order
# to keep week ordering consistent, ensure weeks are in original df.columns order
week_order = {w: i for i, w in enumerate(weeks)}
picks.sort(key=lambda r: week_order[r[0]])

# Output
print("\nSelected picks (Week, Team, Probability):")
for w, t, v in picks:
    print(f"{w:10s} | {t:3s} | {v:.6f}")
print(f"\nTotal objective (sum of selected probabilities): {total:.6f}")
print(f"Number of picks: {len(picks)} of {len(weeks)} weeks (weeks with no eligible >= {threshold} left unassigned).")

# If you want exactly one team per week regardless (even if below threshold),
# change the week constraint from '<= 1' to '== 1' and remove the threshold-based variable creation.
# But per your requirement we do NOT select any prob < {threshold}.