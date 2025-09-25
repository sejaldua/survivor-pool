# Copied solver_historical into src/
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


schedule = pd.read_csv("data/nfl_schedule_2024.csv")

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

df = pd.read_csv('data/team_week_win_prob_2024.csv').set_index('Team').fillna(0)
print(df)

# rest unchanged
print('Done')
