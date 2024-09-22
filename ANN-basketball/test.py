import numpy as np
import pandas as pd
from itertools import combinations

# Load the dataset
data = pd.read_csv('all_seasons.csv')
print(data.columns)

# Convert 'season' to integer year for filtering
data['season'] = data['season'].apply(lambda x: int(x.split('-')[0]))

# Define thresholds for optimal team criteria
top_20_ts = data['ts_pct'].quantile(0.8)
top_10_reb = data['reb'].quantile(0.9)
top_20_rating = data['net_rating'].quantile(0.8)
average_assists = data['ast'].mean()

# Select a pool of 100 players within a 5-year window
pool = data[(data['season'] > 2018) & (data['season'] < 2023)].sample(100)

# Define the features
features = ['ts_pct', 'reb', 'dreb_pct', 'net_rating', 'ast']

# Define a function to evaluate a team
def evaluate_team(team):
    players_ast = [player for _, player in team.iterrows() if float(player['ast']) > average_assists]
    players_ts = [player for _, player in team.iterrows() if float(player['ts_pct']) > top_20_ts]
    players_reb = [player for _, player in team.iterrows() if float(player['reb']) > top_10_reb]
    players_dreb = [player for _, player in team.iterrows() if float(player['dreb_pct']) > 0.2]
    players_rating = [player for _, player in team.iterrows() if float(player['net_rating']) > top_20_rating]
    
    score = 0
    if len(players_ast) >= 2:
        score += 1
    if len(players_ts) >= 2:
        score += 1
    if len(players_reb) >= 1:
        score += 1
    if len(players_dreb) >= 1:
        score += 1
    if len(players_rating) >= 3:
        score += 1
    
    return score

# Find the optimal team
best_team = None
best_score = -1

for team_indices in combinations(pool.index, 5):
    team = pool.loc[list(team_indices)]
    score = evaluate_team(team)
    if score > best_score:
        best_score = score
        best_team = team

# Output the best team
print("Best Team:")
print(best_team[features])