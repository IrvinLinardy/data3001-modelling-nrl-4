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
