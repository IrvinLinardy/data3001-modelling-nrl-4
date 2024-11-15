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




python3 "boost.py"

/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:30: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].mode()[0], inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:30: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].mode()[0], inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:30: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].mode()[0], inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:30: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].mode()[0], inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:30: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].mode()[0], inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
/Users/kevin/Documents/Data3001 week 6 to 10/Xg boost algorithm/boost.py:32: FutureWarning: A value is trying to be set on a copy of a DataFrame or Series through chained assignment using an inplace method.
The behavior will change in pandas 3.0. This inplace method will never work because the intermediate object on which we are setting values always behaves as a copy.

For example, when doing 'df[col].method(value, inplace=True)', try using 'df.method({col: value}, inplace=True)' or df[col] = df[col].method(value) instead, to perform the operation inplace on the original object.


  df[column].fillna(df[column].median(), inplace=True)
Best Parameters: {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200}
Mean Squared Error: 0.8991163370900149
R2 Score: 0.22451272552932644
ZonePossession: 889.0
PossessionSecs: 824.0
RoundId: 496.0
Total_Involved_Tacklers: 374.0
Score: 337.0
SeasonId_2022: 97.0
Set_Type: 87.0
Raw_Tackle_Number_3.0: 83.0
PTB_Contest_Tackled to ground: 82.0
Raw_Tackle_Number_5.0: 80.0
IsHome: 79.0
SeasonId_2023: 75.0
SeasonId_2021: 73.0
PTB_Contest_Stays on feet: 67.0
Club_Id_c14e0139ad91a9741a5731a596aa6549: 67.0
Raw_Tackle_Number_2.0: 66.0
Opposition_Id_837e03d56b4dba3b8a4a5425c0420abd: 63.0
Opposition_Id_5e03a19f4d014a2220665cfd56522d35: 62.0
Club_Id_c03196722c1a837b39f79f1714db475d: 60.0
Raw_Tackle_Number_4.0: 59.0
Opposition_Id_a73752d38e4a78e3e14917f5435ffb6d: 58.0
Opposition_Id_c03196722c1a837b39f79f1714db475d: 54.0
Club_Id_5e03a19f4d014a2220665cfd56522d35: 52.0
Club_Id_b53920c88e4eebf2faa9f4fb43b8944a: 52.0
RunOn: 51.0
Opposition_Id_58485e3acf60682c8fc37d9d521b3019: 51.0
Opposition_Id_c14e0139ad91a9741a5731a596aa6549: 51.0
Club_Id_367ef61d2bc259e608027a8d349c933e: 50.0
Club_Id_3b26834df063f9d51de216a07ec36929: 50.0
PTB_Ultimate_Outcome_Kick: 49.0
Club_Id_a73752d38e4a78e3e14917f5435ffb6d: 47.0
Opposition_Id_367ef61d2bc259e608027a8d349c933e: 47.0
WeatherConditionName_Unknown: 46.0
Opposition_Id_f38f7f087f646c38c0207f1b2af32f12: 46.0
Club_Id_1d6cd83892ee4afdcd8ccd94f817b4a6: 45.0
Club_Id_837e03d56b4dba3b8a4a5425c0420abd: 45.0
Opposition_Id_d3ac47d424b41fd738ec9500dbda2d59: 44.0
Opposition_Id_3b26834df063f9d51de216a07ec36929: 43.0
Club_Id_dc3c7bd8148814b7c4105841baa68e23: 41.0
Club_Id_f38f7f087f646c38c0207f1b2af32f12: 41.0
Club_Id_fdfcde48e2cbf12cc4710a2644b86d85: 40.0
Club_Id_58485e3acf60682c8fc37d9d521b3019: 39.0
Opposition_Id_1d6cd83892ee4afdcd8ccd94f817b4a6: 39.0
Club_Id_d3ac47d424b41fd738ec9500dbda2d59: 38.0
PTB_Ultimate_Outcome_Try: 36.0
Opposition_Id_1d6cd83892ee4afdcd8ccd94f81ftnjhl3s: 36.0
Opposition_Id_980c9c368ae4f1129ea0a6fdd711fa8f: 35.0
Opposition_Id_fdfcde48e2cbf12cc4710a2644b86d85: 35.0
Opposition_Id_b53920c88e4eebf2faa9f4fb43b8944a: 33.0
Raw_Tackle_Number_6.0: 31.0
Opposition_Id_dc3c7bd8148814b7c4105841baa68e23: 29.0
PTB_Ultimate_Outcome_Turnover: 27.0
Club_Id_980c9c368ae4f1129ea0a6fdd711fa8f: 27.0
WeatherConditionName_Rain: 26.0
Club_Id_1d6cd83892ee4afdcd8ccd94f81ftnjhl3s: 22.0
WeatherConditionName_Showers: 20.0
PTB_Ultimate_Outcome_Handover: 20.0
PTB_Ultimate_Outcome_Penalty attack: 19.0
WeatherConditionName_Snow: 18.0
PTB_Ultimate_Outcome_Opp ruck infringement: 18.0
Half_3: 16.0
PTB_Ultimate_Outcome_Other: 15.0
PTB_Ultimate_Outcome_Penalty defence: 13.0
Half_2: 12.0
PTB_Ultimate_Outcome_Field goal attempt: 8.0
Half_4: 7.0
