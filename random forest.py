import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('/Users/kevin/Documents/Data3001 week 6 to 10/ptb_data.csv')

# Drop unneeded columns
df_encoded = df.drop(columns=[
    'PTB Defence', 'Anonymize 1PlayerId', 'EventName', 'PTB Tackle Result',
    'Away Score', 'Home Score', 'MatchId', 'OppPossessionSecs', 'PTB Ultimate Outcome',
    'OppScore', 'Player Id', 'SeqNumber', 'Set', 'TotalPossessionSecs',
    'Tackle', 'ElapsedTime', 'GameTime', 'ZonePhysical', 'OfficialId', 'PositionId'
])

# Standardize column names
df_encoded.columns = df_encoded.columns.str.replace('Raw Tackle Number', 'Raw_Tackle_Number')
df_encoded.columns = df_encoded.columns.str.replace('Club Id', 'Club_Id')
df_encoded.columns = df_encoded.columns.str.replace('Opposition Id', 'Opposition_Id')
df_encoded.columns = df_encoded.columns.str.replace('PTB Contest', 'PTB_Contest')
df_encoded.columns = df_encoded.columns.str.replace('Set Type', 'Set_Type')
df_encoded.columns = df_encoded.columns.str.replace('Total Involved Tacklers', 'Total_Involved_Tacklers')

# Encode categorical variables using pandas get_dummies
df_encoded = pd.get_dummies(df_encoded, columns=['WeatherConditionName', 'Raw_Tackle_Number', 'Half', 
                                                 'PTB_Contest', 'Club_Id', 'Opposition_Id', 'SeasonId'], drop_first=True)

# Handle missing values
df_encoded = df_encoded.dropna()

# Convert boolean columns to integers
df_encoded[df_encoded.select_dtypes(include=['bool']).columns] = df_encoded.select_dtypes(include=['bool']).astype(int)

# Define the numeric features for scaling (excluding the target variable 'DurationSecs')
numeric_features = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
if 'DurationSecs' in numeric_features:
    numeric_features.remove('DurationSecs')

# Setup the Random Forest and GridSearchCV pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('scaler', StandardScaler()),  # Scale features
    ('regressor', RandomForestRegressor(random_state=37))
])

# Define a grid of parameters to search
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

# Setup the grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=2, n_jobs=-1)

# Split the data into training and testing sets
X = df_encoded.drop(['DurationSecs'], axis=1, errors='ignore')  # Adjust the column names based on your needs
y = df_encoded['DurationSecs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Output the best parameters and best score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Feature importances
importances = best_model.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Print the feature importances
print("Feature ranking:")
for i in range(len(features)):
    print(f"{i + 1}. feature {features[indices[i]]} ({importances[indices[i]]:.3f})")


python3 "random forest.py"
Fitting 3 folds for each of 81 candidates, totalling 243 fits
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  44.5s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  44.6s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  49.5s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  49.7s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  49.9s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  45.9s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.7min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.7min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.7min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  41.8s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  42.6s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  42.4s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 2.3min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  44.5s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  43.7s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.4min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.4min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 2.3min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.4min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  44.6s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 2.3min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 2.1min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  43.0s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 2.1min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 2.1min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  43.1s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  44.3s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  41.4s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  41.7s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.2min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.2min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.2min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.5min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  41.8s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  40.1s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 2.2min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 2.2min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  39.6s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.4min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.4min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.4min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 2.2min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  39.9s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  40.3s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  39.8s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 2.1min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 2.1min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 2.1min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.4min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  40.3s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  39.0s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  38.6s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  21.6s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  39.0s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  22.1s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  21.9s
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=200; total time=  44.0s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=200; total time=  43.7s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=200; total time=  43.9s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  22.2s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  22.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  22.0s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=None, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=200; total time=  44.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=200; total time=  44.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  21.8s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=200; total time=  43.7s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  22.1s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  22.6s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  21.9s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=200; total time=  44.1s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  22.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=200; total time=  44.7s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=200; total time=  44.8s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  22.6s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  22.2s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=200; total time=  44.1s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  22.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=200; total time=  45.0s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=200; total time=  44.8s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  22.2s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  22.4s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  22.4s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=200; total time=  44.2s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=200; total time=  44.5s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=200; total time=  45.2s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  22.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  22.1s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  22.1s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=200; total time=  44.1s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=200; total time=  44.6s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=200; total time=  44.7s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  22.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  22.1s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=200; total time=  44.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  22.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=200; total time=  44.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=200; total time=  45.2s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  22.0s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  22.7s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  22.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=200; total time=  44.7s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=200; total time=  44.1s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=200; total time=  44.8s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  22.3s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=200; total time=  44.4s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=200; total time=  44.6s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=200; total time=  44.4s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  39.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  40.1s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  39.5s
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=10, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.1min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  38.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  39.4s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  39.1s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  38.2s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  37.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 2.0min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  37.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  37.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  37.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  39.8s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  38.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=1, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  37.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  38.1s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  36.2s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  36.0s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.3min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  36.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  36.2s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  36.5s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.9min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=100; total time=  36.4s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.8min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  36.4s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.8min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  36.2s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=2, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.8min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=100; total time=  36.9s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.8min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.8min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  36.1s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=2, regressor__n_estimators=300; total time= 1.8min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  35.3s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=200; total time= 1.2min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=100; total time=  34.5s
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.8min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.7min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=5, regressor__n_estimators=300; total time= 1.7min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.1min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.1min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=200; total time= 1.1min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.4min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.4min
[CV] END regressor__max_depth=20, regressor__min_samples_leaf=4, regressor__min_samples_split=10, regressor__n_estimators=300; total time= 1.3min
Best parameters found:  {'regressor__max_depth': 20, 'regressor__min_samples_leaf': 4, 'regressor__min_samples_split': 10, 'regressor__n_estimators': 300}
Best cross-validation score: 0.20
Mean Squared Error: 0.9233551966453682
R^2 Score: 0.20605711951313121
Feature ranking:
1. feature PTB_Contest_Tackled to ground (0.155)
2. feature PossessionSecs (0.155)
3. feature ZonePossession (0.106)
4. feature PTB_Contest_Stays on feet (0.096)
5. feature RoundId (0.076)
6. feature CurrentMargin (0.068)
7. feature Score (0.048)
8. feature Set_Type (0.032)
9. feature Total_Involved_Tacklers (0.031)
10. feature IsHome (0.014)
11. feature SeasonId_2022 (0.012)
12. feature Raw_Tackle_Number_2.0 (0.011)
13. feature SeasonId_2021 (0.010)
14. feature Raw_Tackle_Number_3.0 (0.010)
15. feature SeasonId_2023 (0.008)
16. feature RunOn (0.008)
17. feature WeatherConditionName_Unknown (0.008)
18. feature Raw_Tackle_Number_4.0 (0.008)
19. feature Raw_Tackle_Number_5.0 (0.007)
20. feature Club_Id_3b26834df063f9d51de216a07ec36929 (0.006)
21. feature Club_Id_a73752d38e4a78e3e14917f5435ffb6d (0.006)
22. feature Club_Id_c14e0139ad91a9741a5731a596aa6549 (0.005)
23. feature Opposition_Id_f38f7f087f646c38c0207f1b2af32f12 (0.005)
24. feature Club_Id_b53920c88e4eebf2faa9f4fb43b8944a (0.005)
25. feature Opposition_Id_5e03a19f4d014a2220665cfd56522d35 (0.005)
26. feature Club_Id_367ef61d2bc259e608027a8d349c933e (0.005)
27. feature Opposition_Id_dc3c7bd8148814b7c4105841baa68e23 (0.005)
28. feature Opposition_Id_837e03d56b4dba3b8a4a5425c0420abd (0.005)
29. feature Opposition_Id_c14e0139ad91a9741a5731a596aa6549 (0.005)
30. feature Opposition_Id_3b26834df063f9d51de216a07ec36929 (0.004)
31. feature Club_Id_f38f7f087f646c38c0207f1b2af32f12 (0.004)
32. feature Club_Id_dc3c7bd8148814b7c4105841baa68e23 (0.004)
33. feature Opposition_Id_a73752d38e4a78e3e14917f5435ffb6d (0.004)
34. feature Club_Id_58485e3acf60682c8fc37d9d521b3019 (0.004)
35. feature Club_Id_1d6cd83892ee4afdcd8ccd94f817b4a6 (0.004)
36. feature Opposition_Id_367ef61d2bc259e608027a8d349c933e (0.004)
37. feature Opposition_Id_b53920c88e4eebf2faa9f4fb43b8944a (0.004)
38. feature Opposition_Id_58485e3acf60682c8fc37d9d521b3019 (0.004)
39. feature Club_Id_5e03a19f4d014a2220665cfd56522d35 (0.004)
40. feature Opposition_Id_fdfcde48e2cbf12cc4710a2644b86d85 (0.004)
41. feature Opposition_Id_d3ac47d424b41fd738ec9500dbda2d59 (0.004)
42. feature Opposition_Id_1d6cd83892ee4afdcd8ccd94f817b4a6 (0.004)
43. feature Opposition_Id_c03196722c1a837b39f79f1714db475d (0.004)
44. feature Club_Id_837e03d56b4dba3b8a4a5425c0420abd (0.004)
45. feature Club_Id_d3ac47d424b41fd738ec9500dbda2d59 (0.004)
46. feature Opposition_Id_980c9c368ae4f1129ea0a6fdd711fa8f (0.004)
47. feature Club_Id_c03196722c1a837b39f79f1714db475d (0.004)
48. feature Club_Id_980c9c368ae4f1129ea0a6fdd711fa8f (0.004)
49. feature Club_Id_fdfcde48e2cbf12cc4710a2644b86d85 (0.003)
50. feature Half_2 (0.002)
51. feature Opposition_Id_1d6cd83892ee4afdcd8ccd94f81ftnjhl3s (0.002)
52. feature WeatherConditionName_Showers (0.002)
53. feature WeatherConditionName_Rain (0.001)
54. feature Raw_Tackle_Number_6.0 (0.001)
55. feature Club_Id_1d6cd83892ee4afdcd8ccd94f81ftnjhl3s (0.001)
56. feature WeatherConditionName_Snow (0.000)
57. feature Half_3 (0.000)
58. feature Half_4 (0.000)
59. feature Raw_Tackle_Number_7.0 (0.000)
60. feature PTB_Contest_Other (0.000)
