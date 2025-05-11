import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from scipy.stats import randint

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Grid Search
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Define the parameter distribution for Random Search
param_dist = {
    'n_estimators': randint(10, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10)
}

# Initialize models
rf = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Random Search
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=50, cv=5, random_state=42)
random_search.fit(X_train, y_train)

# Evaluate the models
grid_accuracy = accuracy_score(y_test, grid_search.predict(X_test))
random_accuracy = accuracy_score(y_test, random_search.predict(X_test))

# Plot comparison of accuracies
methods = ['Grid Search', 'Random Search']
accuracies = [grid_accuracy, random_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(methods, accuracies, color=['blue', 'green'])
plt.title('Comparison of Grid Search and Random Search Performance', fontsize=14)
plt.xlabel('Search Method', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim([0, 1])
plt.tight_layout()

# Display the plot
plt.show()
