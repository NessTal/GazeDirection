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
    X[column+'x'] = [point[0] for point in data[column]]
    X[column+'y'] = [point[1] for point in data[column]]

# Get labels
Y = data['Code']

# Split into train and test sets, keeping each participant's data in the same set to prevent data leakage
participants = X['ParticipantID'].unique()
train_participants, test_participants = train_test_split(participants, test_size=0.2, random_state=42)

X_train = X[X['ParticipantID'].isin(train_participants)].drop(columns=['ParticipantID'])
X_test = X[X['ParticipantID'].isin(test_participants)].drop(columns=['ParticipantID'])
Y_train = Y[X['ParticipantID'].isin(train_participants)]
Y_test = Y[X['ParticipantID'].isin(test_participants)]

