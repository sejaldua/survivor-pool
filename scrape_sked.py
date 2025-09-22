import pandas as pd
import requests
from bs4 import BeautifulSoup
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Scrape the NFL schedule for the current season
url = "https://www.pro-football-reference.com/years/2024/games.htm"
response = requests.get(url)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

# Find the schedule table
table = soup.find("table", id="games")

# Extract headers
headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]

# Extract rows
rows = []
for row in table.find("tbody").find_all("tr"):
    if "class" in row.attrs and "thead" in row["class"]:
        continue  # skip mid-table headers
    cells = [cell.get_text(strip=True) for cell in row.find_all(["th", "td"])]
    rows.append(cells)

# Create DataFrame
sked_df = pd.DataFrame(rows, columns=headers)
sked_df.columns = ['Week', 'Day', 'Date', 'Time', 'Team1', 'H/A', 'Team2', 'Boxscore', 'PtsW', 'PtsL', 'YdsW', 'TOW', 'YdsL', 'TOL']
sked_df = sked_df[['Week', 'Day', 'Date', 'Time', 'Team1', 'H/A', 'Team2']]
sked_df['Home'] = sked_df.apply(lambda x: x['Team2'] if x['H/A'] == '@' else x['Team1'], axis=1)
sked_df['Away'] = sked_df.apply(lambda x: x['Team1'] if x['H/A'] == '@' else x['Team2'], axis=1)
sked_df = sked_df[sked_df['Week'].isin(map(str, range(1, 19)))]
sked_df['Date'] = pd.to_datetime(sked_df['Date'])
sked_df.to_csv('nfl_schedule_2024.csv', index=False)