import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
import pandas as pd

def extract_features(player):
    return {
        'name': player[1],
        'ts_pct': float(player[19]),
        'reb': float(player[13]),
        'dreb_pct': float(player[17]),
        'rating': float(player[15]),
        'ast': float(player[14])
    }

def team_score(team):
    ts_pct = np.mean([player['ts_pct'] for player in team])
    reb = np.sum([player['reb'] for player in team])
    dreb_pct = np.max([player['dreb_pct'] for player in team])
    rating = np.mean([player['rating'] for player in team])
    ast = np.sum([player['ast'] for player in team])
    
    score = (ts_pct * 20 + reb * 0.5 + dreb_pct * 10 + rating + ast * 0.5) / 5
    return score

def select_optimal_team(pool, team_size=5):
    player_features = [extract_features(player) for player in pool]
    
    best_team = None
    best_score = -float('inf')
    
    # Use a heuristic to find a good solution
    for _ in range(1000):  # Number of iterations
        team = random.sample(player_features, team_size)
        score = team_score(team)
        if score > best_score:
            best_score = score
            best_team = team
    
    return best_team, best_score

# Load the dataset
data = pd.read_csv('all_seasons.csv')
print(data.columns)

# Convert 'season' to integer year for filtering
data['season'] = data['season'].apply(lambda x: int(x.split('-')[0]))

# Select a pool of 100 players within a 5-year window
pool = data[(data['season'] > 2018) & (data['season'] < 2023)].sample(100).values.tolist()

# Select the optimal team from the pool
optimal_team, optimal_score = select_optimal_team(pool)

# Output the optimal team and its score
print("Optimal Team:")
for player in optimal_team:
    print(f"Name: {player['name']}, TS%: {player['ts_pct']:.3f}, REB: {player['reb']:.1f}, DREB%: {player['dreb_pct']:.3f}, Rating: {player['rating']:.1f}, AST: {player['ast']:.1f}")
print(f"\nTeam Score: {optimal_score:.2f}")