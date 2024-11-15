import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Load data
df = pd.read_csv('/Users/kevin/Documents/Data3001 week 6 to 10/ptb_data.csv')

# Drop columns that may not be necessary or relevant
df.drop(columns=[
    'PTB Defence', 'Anonymize 1PlayerId', 'EventName', 'PTB Tackle Result',
    'Away Score', 'Home Score', 'MatchId', 'OppPossessionSecs',
    'OppScore', 'Player Id', 'SeqNumber', 'Set', 'TotalPossessionSecs', 
    'Tackle', 'ElapsedTime', 'GameTime', 'ZonePhysical', 'OfficialId', 'PositionId'
], inplace=True)

# Rename columns for consistency
rename_dict = {
    'Raw Tackle Number': 'Raw_Tackle_Number', 'Club Id': 'Club_Id',
    'Opposition Id': 'Opposition_Id', 'PTB Ultimate Outcome': 'PTB_Ultimate_Outcome',
    'PTB Contest': 'PTB_Contest', 'Set Type': 'Set_Type',
    'Total Involved Tacklers': 'Total_Involved_Tacklers'
}
df.rename(columns=rename_dict, inplace=True)

# Fill missing values
for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)

# Create dummy variables
categorical_vars = ['WeatherConditionName', 'Raw_Tackle_Number', 'Half', 'PTB_Ultimate_Outcome', 'PTB_Contest', 'Club_Id', 'Opposition_Id', 'SeasonId']
df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)

# Convert boolean columns to integer
df[df.select_dtypes(include=['bool']).columns] = df.select_dtypes(include=['bool']).astype(int)

# Prepare the data for modeling
target = 'DurationSecs'
X = df.drop(['CurrentMargin', target], axis=1)  # Ensure CurrentMargin is not part of the input if not relevant
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train XGBoost Model with grid search for basic hyperparameter tuning
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
parameters = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}
grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Predictions and Evaluation
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Print feature importance
importance = best_model.get_booster().get_score(importance_type='weight')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importance:
    print(f'{feature}: {importance}')
