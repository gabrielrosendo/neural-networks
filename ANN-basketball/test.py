import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

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

# Define a function to label the data
def is_optimal_team(team):
    label = 0
    players_ast = [player for _, player in team.iterrows() if float(player['ast']) > average_assists]
    if len(players_ast) >= 2:
        players_ts = [player for _, player in team.iterrows() if float(player['ts_pct']) > top_20_ts]
        if len(players_ts) >= 2:
            players_reb = [player for _, player in team.iterrows() if float(player['reb']) > top_10_reb]
            if len(players_reb) >= 1:
                players_dreb = [player for _, player in team.iterrows() if float(player['dreb_pct']) > 0.2]
                if len(players_dreb) >= 1:
                    players_rating = [player for _, player in team.iterrows() if float(player['net_rating']) > top_20_rating]
                    if len(players_rating) >= 3:
                        label = 1
    return label

# Select a pool of 100 players within a 5-year window
pool = data[(data['season'] > 2018) & (data['season'] < 2023)].sample(100)

# Define the features and labels
features = ['ts_pct', 'reb', 'dreb_pct', 'net_rating', 'ast']
X = []
y = []

for i in range(0, len(pool), 5):
    team = pool.iloc[i:i+5]
    if len(team) == 5:
        team_features = team[features].values.flatten()
        X.append(team_features)
        y.append(is_optimal_team(team))

X = np.array(X)
y = np.array(y)

# Normalize the features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Output shapes
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Output number of y = 1 occurrences
count_ones = np.sum(y == 1)
print(f"Number of 1's in y: {count_ones}")

# Resample the dataset to address class imbalance
X = pd.DataFrame(X)
y = pd.Series(y)

# Separate majority and minority classes
X_majority = X[y == 0]
X_minority = X[y == 1]
y_majority = y[y == 0]
y_minority = y[y == 1]

# Upsample minority class
X_minority_upsampled, y_minority_upsampled = resample(X_minority, y_minority,
                                                      replace=True,  # sample with replacement
                                                      n_samples=len(X_majority),  # to match majority class
                                                      random_state=42)  # reproducible results

# Combine majority class with upsampled minority class
X_upsampled = pd.concat([X_majority, X_minority_upsampled])
y_upsampled = pd.concat([y_majority, y_minority_upsampled])

# Convert back to numpy arrays
X = X_upsampled.values
y = y_upsampled.values

# Output number of y = 1 occurrences after resampling
count_ones = np.sum(y == 1)
print(f"Number of 1's in y after resampling: {count_ones}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network architecture
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.01
epochs = 1000

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.01
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.01
b2 = np.zeros((1, output_size))

# Activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Forward propagation
def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Compute the cost
def compute_cost(A2, y):
    m = y.shape[0]
    cost = -np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2)) / m
    return cost

# Backpropagation with gradient clipping
def backward_propagation(X, y, Z1, A1, Z2, A2):
    m = y.shape[0]
    dZ2 = A2 - y.reshape(-1, 1)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Gradient clipping
    clip_value = 1.0
    dW1 = np.clip(dW1, -clip_value, clip_value)
    db1 = np.clip(db1, -clip_value, clip_value)
    dW2 = np.clip(dW2, -clip_value, clip_value)
    db2 = np.clip(db2, -clip_value, clip_value)

    return dW1, db1, dW2, db2

# Update weights and biases
def update_parameters(dW1, db1, dW2, db2):
    global W1, b1, W2, b2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Train the model
for epoch in range(epochs):
    Z1, A1, Z2, A2 = forward_propagation(X_train)
    cost = compute_cost(A2, y_train)
    dW1, db1, dW2, db2 = backward_propagation(X_train, y_train, Z1, A1, Z2, A2)
    update_parameters(dW1, db1, dW2, db2)
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Cost: {cost}')

# Evaluate the model
_, _, _, A2_test = forward_propagation(X_test)
y_pred = (A2_test > 0.5).astype(int)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Explanation of the architecture
print("""
The architecture of the neural network consists of:
1. Input Layer: 25 features (5 features for each of the 5 players).
2. Hidden Layer: 64 neurons with sigmoid activation.
3. Output Layer: 1 neuron with sigmoid activation for binary classification.

The input features for each player include:
- True Shooting Percentage (ts_pct)
- Rebounds (reb)
- Defensive Rebound Percentage (dreb_pct)
- Net Rating (net_rating)
- Assists (ast)

The model is trained to predict whether a team is optimal (1) or not (0) based on these features.
""")

# Interpretation of the output
print("""
The output of the MLP is interpreted as follows:
- A predicted value close to 1 indicates that the team is optimal.
- A predicted value close to 0 indicates that the team is not optimal.

Based on the trained model, you can now use it to predict the optimality of new teams by providing the features of the 5 players.
""")