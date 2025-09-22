import pulp
import numpy as np
from collections import defaultdict
from itertools import product
import random
import pandas as pd

mapping = {
    'Arizona Cardinals': 'ARI',
    'Atlanta Falcons': 'ATL',
    'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF',
    'Carolina Panthers': 'CAR',
    'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN',
    'Cleveland Browns': 'CLE',
    'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN',
    'Detroit Lions': 'DET',
    'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU',
    'Indianapolis Colts': 'IND',
    'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC',
    'Las Vegas Raiders': 'LV',
    'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LAR',
    'Miami Dolphins': 'MIA',
    'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE',
    'New Orleans Saints': 'NO',
    'New York Giants': 'NYG',
    'New York Jets': 'NYJ',
    'Philadelphia Eagles': 'PHI',
    'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF',
    'Seattle Seahawks': 'SEA',
    'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN',
    'Washington Commanders': 'WAS',
}


schedule = pd.read_csv("nfl_schedule_2024.csv")

# Create lookup: (team, week) -> True if home, False if away
is_home_lookup = {}
for _, row in schedule.iterrows():
    is_home_lookup[(mapping[row["Home"]], f"Week {row['Week']}")] = True
    is_home_lookup[(mapping[row["Away"]], f"Week {row['Week']}")] = False

def is_home(team, week):
    return is_home_lookup.get((team, week), False)

# Assume schedule has columns: Week, Home, Away
matchup_lookup = {}
for _, row in schedule.iterrows():
    week = f"Week {row['Week']}"
    home = mapping[row["Home"]]
    away = mapping[row["Away"]]

    matchup_lookup[(home, week)] = f"{away} @ {home}"
    matchup_lookup[(away, week)] = f"{away} @ {home}"

df = pd.read_csv('team_week_win_prob_2024.csv').set_index('Team').fillna(0)
print(df)

# Parameters
threshold = 0.70  # only allow picks with prob >= threshold

teams = list(df.index)
weeks = list(df.columns)

# Create LP problem
prob_lp = pulp.LpProblem("Pick_Teams_Per_Week", pulp.LpMaximize)

# Decision variables: only create variable if prob >= threshold and not NaN
x = {}
for t in teams:
    for w in weeks:
        val = df.loc[t, w]
        if pd.notna(val) and val >= threshold and is_home(t, w):
            var_name = f"x_{t}_{w}"
            x[(t, w)] = pulp.LpVariable(var_name, cat='Binary')
# Constraints:
# 1) Exactly one team per week
for w in weeks:
    vars_this_week = [x[(t, w)] for t in teams if (t, w) in x]
    prob_lp += pulp.lpSum(vars_this_week) == 1, f"ExactlyOneTeam_Week_{w}"

# 2) Each team can be selected at most once across all weeks
for t in teams:
    vars_this_team = [x[(t, w)] for w in weeks if (t, w) in x]
    if vars_this_team:
        prob_lp += pulp.lpSum(vars_this_team) <= 1, f"AtMostOneWeek_Team_{t}"

# Objective: maximize sum of selected probabilities
objective_terms = []
for (t, w), var in x.items():
    base_prob = df.loc[t, w] 
    # if is_home(t, w):
    #     objective_terms.append((base_prob + 0.01) * var)  # slight bonus for home teams
    # else:
    objective_terms.append(base_prob * var)
    
prob_lp += pulp.lpSum(objective_terms)
prob_lp += x[('CIN', 'Week 1')] != 1

# Solve
solver = pulp.PULP_CBC_CMD(msg=False)  # change msg=True to get solver log
prob_lp.solve(solver)

# Collect solution
picks = []
total = []
for (t, w), var in x.items():
    if pulp.value(var) == 1:
        matchup = matchup_lookup.get((t, w), "Unknown")
        picks.append((w, t, float(df.loc[t, w]), matchup))
        total.append(float(df.loc[t, w]))


# Sort picks by week order
week_order = {w: i for i, w in enumerate(weeks)}
picks.sort(key=lambda r: week_order[r[0]])

# Create a DataFrame from the solution
solution_dict = {
    'Week': [p[0] for p in picks],
    'Team': [p[1] for p in picks],
    'Probability': [p[2] for p in picks],
    'Matchup': [p[3] for p in picks]
}
df_solution = pd.DataFrame(solution_dict)

# Export the DataFrame to a CSV file
output_filename = 'survivor_picks_2024.csv'
df_solution.to_csv(output_filename, index=False)

print(f"The solver output has been successfully converted into a DataFrame and exported to '{output_filename}'.")
print("\nDataFrame Preview:")
print(df_solution)

print(f"\nTotal survival probability: {np.prod(total)*100:.3f}%")
