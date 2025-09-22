import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib import font_manager

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

def generate_team_week_figure(team_week_csv='team_week_win_prob.csv', schedule_csv='nfl_schedule_2025.csv', out_png='team_week_grid.png'):
    """Generate a matplotlib figure (teams x weeks) where each cell shows the matchup (AWAY@HOME)
    and that team's win probability. Cells are color-coded by probability and saved to a PNG.
    Uses Commissioner if available, otherwise falls back to DejaVu Sans.
    """

    # read data
    probs = pd.read_csv(team_week_csv, index_col=0)
    weeks = sorted([c for c in probs.columns if c.lower().startswith('week')], key=lambda x: int(x.split()[-1]))
    probs = probs.reindex(columns=weeks)

    sched = pd.read_csv(schedule_csv)
    if 'Home_Abbrev' not in sched.columns or 'Away_Abbrev' not in sched.columns:
        sched['Home_Abbrev'] = sched['Home'].map(mapping)
        sched['Away_Abbrev'] = sched['Away'].map(mapping)

    teams = sorted(probs.index)
    n_teams = len(teams)
    n_weeks = len(weeks)

    # Build numeric and display arrays
    numeric = np.full((n_teams, n_weeks), np.nan)
    display = np.full((n_teams, n_weeks), '', dtype=object)
    team_to_idx = {t: i for i, t in enumerate(teams)}

    for j, w in enumerate(weeks):
        week_num = int(w.split()[-1])
        week_df = sched[sched['Week'] == week_num]
        for _, row in week_df.iterrows():
            home = row['Home_Abbrev']
            away = row['Away_Abbrev']
            matchup = f"{away}@{home}"
            home_p = None
            away_p = None
            if home in probs.index and w in probs.columns:
                try:
                    home_p = float(probs.loc[home, w])
                except Exception:
                    home_p = np.nan
            if away in probs.index and w in probs.columns:
                try:
                    away_p = float(probs.loc[away, w])
                except Exception:
                    away_p = np.nan

            if home in team_to_idx:
                i = team_to_idx[home]
                numeric[i, j] = home_p
                display[i, j] = f"{matchup}\n{'' if np.isnan(home_p) else f'{home_p*100:.1f}%'}"
            if away in team_to_idx:
                i = team_to_idx[away]
                numeric[i, j] = away_p
                display[i, j] = f"{matchup}\n{'' if np.isnan(away_p) else f'{away_p*100:.1f}%'}"

    try:
        funnel_font = font_manager.findfont('IBM Plex Sans', fallback_to_default=False)
        font_manager.fontManager.addfont(funnel_font)
        plt.rcParams['font.family'] = 'IBM Plex Sans'
    except Exception:
        plt.rcParams['font.family'] = 'DejaVu Sans'

    # Prepare colormap: red (low) -> green (high)
    cmap = plt.get_cmap('RdYlGn')
    norm = colors.Normalize(vmin=0.0, vmax=1.0)

    # figure size: 0.6 inch per column, 0.25 inch per row minimum
    cell_w = 0.9
    cell_h = 0.28
    fig_w = max(8, n_weeks * cell_w)
    fig_h = max(6, n_teams * cell_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # background grid and ticks
    ax.set_xlim(0, n_weeks)
    ax.set_ylim(0, n_teams)
    ax.invert_yaxis()

    ax.tick_params(axis='x', which='major', pad=6)
    ax.tick_params(axis='both', which='both', length=6, color='#333333')

    # draw cells
    for i in range(n_teams):
        for j in range(n_weeks):
            v = numeric[i, j]
            x = j
            y = i
            if np.isnan(v):
                color = '#efefef'
                text_col = '#888'
            else:
                # normalized between 0..1
                nv = max(0.0, min(1.0, float(v)))
                color = cmap(norm(nv))
                text_col = 'black' if nv > 0.5 else 'white'
            rect = plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='white')
            ax.add_patch(rect)
            # text: matchup and prob on two lines
            txt = display[i, j]
            if txt:
                # center text vertically and horizontally; use contrast-aware color
                # determine text color based on luminance
                lines = txt.split('\n')
                # color may be RGBA tuple; convert to rgb
                try:
                    r, g, b = (color[0], color[1], color[2]) if isinstance(color, tuple) else colors.to_rgb(color)
                except Exception:
                    r, g, b = 1.0, 1.0, 1.0
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                text_col = 'black' if lum > 0.5 else 'white'
                ax.text(x + 0.5, y + 0.5, '\n'.join(lines), ha='center', va='center', fontsize=7, color=text_col)

    ax.set_frame_on(False)
    # major ticks at centers (with labels)
    ax.set_xticks(np.arange(n_weeks) + 0.5)
    ax.set_xticklabels(weeks, fontsize=7)
    ax.set_yticks(np.arange(n_teams) + 0.5)
    ax.set_yticklabels(teams, fontsize=7)


    # grid only on minors
    ax.grid(which="minor", color="#cccccc", linewidth=0.6)
    ax.xaxis.set_ticks_position('top')
    ax.set_axisbelow(False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'Wrote {out_png}')


if __name__ == '__main__':
    generate_team_week_figure()