import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

data_file = 'landmarks_and_hand_coding_filtered.parquet'

def load_data_and_split_train_test(data_file,test_size=0.2):
    # Load data
    data = pd.read_parquet(data_file)
    data['ParticipantID'] = [file.split('_')[0] for file in data['File']]
    #data = data[data['Code'] != 'x'].copy()

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
    train_participants, test_participants = train_test_split(participants, test_size=test_size, random_state=42)

    X_train = X[X['ParticipantID'].isin(train_participants)].drop(columns=['ParticipantID'])
    X_test = X[X['ParticipantID'].isin(test_participants)].drop(columns=['ParticipantID'])
    Y_train = Y[X['ParticipantID'].isin(train_participants)]
    Y_test = Y[X['ParticipantID'].isin(test_participants)]

    Y_train['Code'] = Y_train['Code'].astype('category')
    Y_test['Code'] = Y_test['Code'].astype('category')

    # Save train and test sets
    X_train.to_parquet('X_train.parquet')
    X_test.to_parquet('X_test.parquet')
    Y_train.to_parquet('Y_train.parquet')
    Y_test.to_parquet('Y_test.parquet')

    return X_train, X_test, Y_train, Y_test

#X_train, X_test, Y_train, Y_test = load_data_and_split_train_test(data_file)

def xgboost_with_grid_search(X_train, X_test, Y_train, Y_test, parameters):
    # Encode labels
    Y_train = LabelEncoder().fit_transform(Y_train)
    Y_test = LabelEncoder().fit_transform(Y_test)

    # Set up classifier
    xgb_model = xgb.XGBClassifier(objective="multi:softmax",random_state=42)

    # Run grid search
    grid_search = GridSearchCV(xgb_model, parameters, scoring='accuracy', n_jobs=6, cv=5, verbose=3)
    grid_search.fit(X_train[3000:5000], Y_train[3000:5000])

    # Print best parameters
    print(grid_search.best_params_)
    print(grid_search.best_score_)

    # Save model
    grid_search.best_estimator_.save_model('xgb_model.json')

    # Predict on test set
    Y_pred = grid_search.predict(X_test[13000:15000])

    # Print accuracy
    print('Accuracy: '+str(np.mean(Y_pred == Y_test[13000:15000])))

    return grid_search, Y_pred


if __name__ == '__main__':
    # Load train and test sets
    X_train = pd.read_parquet('X_train.parquet')
    X_test = pd.read_parquet('X_test.parquet')
    Y_train = pd.read_parquet('Y_train.parquet')
    Y_test = pd.read_parquet('Y_test.parquet')

    # Set up parameters for grid search   
    parameters = {
    'max_depth': range(2, 10, 1),
    'n_estimators': range(60, 220, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}
    grid_search, Y_pred = xgboost_with_grid_search(X_train, X_test, Y_train, Y_test, parameters)
