import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_parquet('landmarks_and_hand_coding_filtered.parquet')
data['ParticipantID'] = [file.split('_')[0] for file in data['File']]

# Get features
X = pd.DataFrame()
X['ParticipantID'] = data['ParticipantID']
for column in data.columns[2:-2]:
    X[column+'_x'] = [point[0] for point in data[column]]
    X[column+'_y'] = [point[1] for point in data[column]]
X = X.copy()

# Get labels
Y = data[['Code']].copy()

# Split into train and test sets, keeping each participant's data in the same set to prevent data leakage
participants = X['ParticipantID'].unique()
train_participants, test_participants = train_test_split(participants, test_size=0.2, random_state=42)

X_train = X[X['ParticipantID'].isin(train_participants)].drop(columns=['ParticipantID'])
X_test = X[X['ParticipantID'].isin(test_participants)].drop(columns=['ParticipantID'])
Y_train = Y[X['ParticipantID'].isin(train_participants)]
Y_test = Y[X['ParticipantID'].isin(test_participants)]

# save train and test sets
X_train.to_parquet('X_train.parquet')
X_test.to_parquet('X_test.parquet')
Y_train.to_parquet('Y_train.parquet')
Y_test.to_parquet('Y_test.parquet')

# Set up parameters for grid search
parameters = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'n_estimators': [100, 250, 500, 750, 1000],
                'gamma': [0, 0.25, 0.5, 1.0],
                'min_child_weight': [1, 5, 10, 15, 20],
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],
                'reg_alpha': [0, 0.25, 0.5, 1.0],
                'reg_lambda': [0.2, 0.4, 0.6, 0.8, 1.0],
                'scale_pos_weight': [0.2, 0.4, 0.6, 0.8, 1.0]}

# Set up classifier
xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)

# Run grid search
grid_search = GridSearchCV(xgb_model, parameters, scoring='accuracy', n_jobs=-1, cv=5, verbose=3)
grid_search.fit(X_train[:100], Y_train[:100])

# Print best parameters
print(grid_search.best_params_)
print(grid_search.best_score_)

# Save model
grid_search.best_estimator_.save_model('xgb_model.json')

# Predict on test set
Y_pred = grid_search.predict(X_test)

# Print accuracy
print('Accuracy: '+str(np.mean(Y_pred == Y_test)))