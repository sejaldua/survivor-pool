import pandas as pd
import math

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

df = pd.read_csv('nfl_schedule_2025.csv')
print(df.head(40))
ratings_df = pd.read_csv('ratings.csv').set_index('Team').to_dict(orient='index')

def compute_win_prob(home_rating, away_rating, hfa=0.25):
    delta = (home_rating - away_rating) + hfa
    return 1.0 / (1.0 + math.exp(-delta))

df['Home_Rating'] = df['Home'].apply(lambda x: ratings_df[x]['Rating'])
df['Away_Rating'] = df['Away'].apply(lambda x: ratings_df[x]['Rating'])
df['Win_Prob_Home'] = df.apply(lambda x: compute_win_prob(x['Home_Rating'], x['Away_Rating']), axis=1)
print(df[df['Week'] == 3].sort_values(by='Win_Prob_Home', ascending=False).head(20))
df['Win_Prob_Away'] = 1 - df['Win_Prob_Home']
df['Max_WP'] = df[['Win_Prob_Home', 'Win_Prob_Away']].max(axis=1)
df['Home_Abbrev'] = df['Home'].apply(lambda x: mapping[x])
df['Away_Abbrev'] = df['Away'].apply(lambda x: mapping[x])

# create a table where rows are teams and columns are weeks with win prob for each team
team_week_wp = pd.DataFrame()
for week in df['Week'].unique():
    week_df = df[df['Week'] == week]
    for _, row in week_df.iterrows():
        team_week_wp = pd.concat([team_week_wp, pd.DataFrame({'Team': [row['Home_Abbrev']], f'Week {week}': [row['Win_Prob_Home']]})], axis=0)
        team_week_wp = pd.concat([team_week_wp, pd.DataFrame({'Team': [row['Away_Abbrev']], f'Week {week}': [row['Win_Prob_Away']]})], axis=0)
team_week_wp = team_week_wp.groupby('Team').max().reset_index()
team_week_wp = team_week_wp.set_index('Team').sort_index()
team_week_wp.to_csv('team_week_win_prob.csv')


# make a table where each week is a column and each row is a mathcup with win prob for home team sorted in descending order
# the index should be arbitrary matchup numbers
# weeks = df['Week'].unique()
# weeks.sort()
# table = pd.DataFrame()
# for week in weeks:
#     week_df = df[df['Week'] == week].copy()
#     week_df = week_df[['Home_Abbrev', 'Away_Abbrev', 'Max_WP']]
#     week_df = week_df.sort_values(by='Max_WP', ascending=False).reset_index(drop=True)
#     week_df.index = week_df.index + 1
#     week_df['Max_WP'] = week_df.apply(lambda x: f"{x['Away_Abbrev']} @ {x['Home_Abbrev']} : {round(x['Max_WP'], 3)}", axis=1)
#     week_df.columns = ['Home', 'Away', f'Week {week}']
#     table = pd.concat([table, week_df[[f'Week {week}']]], axis=1)

# table.to_csv('cheatsheet.csv')