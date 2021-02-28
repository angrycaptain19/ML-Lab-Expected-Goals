# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 08:32:03 2020

@author: Jannik Spiess
"""

#### Preparation ##############################################################


# Import packages
import json as json
import pandas as pd
import numpy as np
import glob as glob
import sklearn
import tensorflow as tf
import keras
import matplotlib
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

# Import json. files with event data (Files can be downloaded from Statsbomb
# here: https://github.com/statsbomb/open-data)

# Initialize list
events = [] 

# Import all json files from current directory
events_all = [i for i in glob.glob('*.json')] 
for i in events_all :
    event = json.load(open(i, encoding="utf8"))
    events.append(event)

# Remark: Loading all json files at once can cause memory problems. If memory 
# problems occur, the files can be split into multiple components 
# (e.g. ~ 100 files per component), extract the shots for each component 
# individually and concatenate the resulting dataframes as described in lines 67-76 


#### Data Pre-Processing ######################################################

# Initialize dataframe
events_df = pd.DataFrame() 

# Generate dataframe containing all events
for i in events :
    df = pd.DataFrame(i)
    events_df = events_df.append(df, ignore_index=True, sort=False)

# Extract relevant variables by subsetting needed columns
events_df = events_df.loc[:, ['play_pattern', 'location', 'under_pressure', 'shot']]

# Extract shots by omitting all rows without an entry for the 'shot' variable
shots_df = events_df.dropna(subset=['shot'])


# Optional: Concatenate single shot dataframes (lines 67-76)

# Initialize dataframe
#shots_df_all = pd.DataFrame()

# Append dataframe from individual component
#shots_df_all = shots_df_all.append(shots_df, ignore_index=True, sort=False) 

# After concatenating all dataframes store as shots_df to continue with script
#shots_df = shots_df_all


# Transform object variables into individual variables (Shot variable)
shots_df = pd.concat([shots_df, shots_df['shot'].apply(pd.Series)], axis=1).drop(['shot'], axis=1) 

# Transform object variables into individual variables (Remaining variables)
object_variables = ['play_pattern', 'outcome', 'type', 'technique', 'body_part'] 
for i in object_variables :
    shots_df = pd.concat([shots_df, shots_df[i].apply(pd.Series)], 
    axis=1).drop(['id', i], axis=1).rename(columns={'name': i})

# Omit unneeded variables
shots_df = shots_df.drop(['statsbomb_xg', 'end_location', 'key_pass_id', 'deflected', 
                          'redirect', 'saved_off_target', 'saved_to_post', 'kick_off'], axis=1)
 
# Omit observations with na values in the freeze frame
shots_df = shots_df.dropna(subset=['freeze_frame'])
    
# Transform non-binary target variable into binary target variable with value 1
# if the outcome is a goal and 0 otherwise
shots_df['outcome'] = np.where(shots_df['outcome'] == 'Goal', 1, 0)

# Transform non-binary independent variables into binary dummy variables
non_binary_variables = ['play_pattern', 'type', 'technique', 'body_part']
for i in non_binary_variables :
    groups = shots_df.groupby(i).groups.keys()
    for j in groups:
        shots_df[j] = np.where(shots_df[i] == j, True, False)
    shots_df = shots_df.drop([i], axis=1)

# Transform binary independent variables into equivalent format
binary_variables = ['under_pressure', 'follows_dribble', 'first_time', 
                    'one_on_one', 'aerial_won', 'open_goal']
for i in binary_variables :
        shots_df[i] = np.where(shots_df[i] == True, True, False)


#### Feature Engineering ######################################################

# 1) Extract spatial features from location variable 
# (distance to goal and angle to goal)

# Reset index of shots_df
shots_df = shots_df.reset_index(drop=True)
# Extract location variable from shots_df
shots_df_location = shots_df.loc[:, ['location']]
# Transform two-dimensional location array into individual variables
shots_df_location = shots_df_location['location'].apply(pd.Series)
# Iterate over all shots
for i, row in shots_df_location.iterrows():
    # Compute x distance to the goal line
    x = 120 - shots_df_location.iloc[i, 0]
    # Compute y distance to the middle of the field
    y = abs(shots_df_location.iloc[i, 1] - 40)
    # Compute distance d to the middle of the goal
    d = np.sqrt(x**2 + y**2)
    # Compute  goal angle a (in radians)
    a = np.arctan(8.00 *x /(x**2 + y**2 - (8.00/2)**2)) 
    if a < 0 :
        a = np.pi + a
    # Store distance to the goal as variable
    shots_df.at[i,'distance to goal'] = d
    # Store angle to the goal as variable
    shots_df.at[i, 'angle to goal'] = a


# 2) Extract positional features from freeze-frame variable 
# (distance to nearest opponent and distance to goalkeeper)

# Extract freeze frame from shots_df
shots_df_freeze = shots_df.loc[:, ['freeze_frame']] 
# Transform freeze frame into individual variables
shots_df_freeze = shots_df_freeze['freeze_frame'].apply(pd.Series)
# Initialize list of columns
columns = list(shots_df_freeze)
# Iterate over all shots
for i, row in shots_df_freeze.iterrows() :
    # Extract x and y coordinates of the shot
    x1 = shots_df_location.iloc[i, 0]
    y1 = shots_df_location.iloc[i, 1]
    # Initialize list
    distance_opp = []
    # Iterate over all players in the freeze frame
    for j in columns :
        # Ignore cells with na. values
        if pd.isna(shots_df_freeze.iloc[i, j]) == False :
            # Subset opponents 
            if shots_df_freeze.iloc[i, j]['teammate'] == False :
                # Extract x and coordinate of opponent
                x2 = shots_df_freeze.iloc[i, j]['location'][0] 
                y2 = shots_df_freeze.iloc[i, j]['location'][1] 
                # Compute euclidean distance d to opponent
                d = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Append distance d to list
                distance_opp.append(d)
                # Subset distance to the goalkeeper
                if shots_df_freeze.iloc[i, j]['position']['name'] == 'Goalkeeper' :
                    distance_gk = d
    if not distance_opp:
        shots_df.at[i,'min distance opponents'] = np.nan
        shots_df.at[i, 'distance goalkeeper'] = np.nan
    else :
        # Store minimum distance to opponent as variable
        shots_df.at[i,'min distance opponents'] = min(distance_opp)
        # Store distance to the goalkeeper as variable
        shots_df.at[i, 'distance goalkeeper'] = distance_gk


# 3) Extract positional features from freeze-frame variable 
# (Number of teammates / opponents within shot angle)

# Pre-define coordinates of the pitch as given by the documentation
p1x = 120 # x coordinate of right post of the goal
p1y = 36 # y coordinate of right post of the goal
p2x = 120 # x coordinate of left post of the goal
p2y = 44 # y coordinate of left post of the goal
# Iterate over all shots
for i, row in shots_df_freeze.iterrows() :
    # Extract x and y coordinates of the shot
    p0x = shots_df_location.iloc[i, 0] 
    p0y = shots_df_location.iloc[i, 1]
    # Initialize lists
    teammates = []
    opponents = []
    # Iterate over all players in the freeze frame 
    for j in columns :
        # Ignore cells with na. values
        if pd.isna(shots_df_freeze.iloc[i, j]) == False : 
            # Extract x and y coordinates of player
            px = shots_df_freeze.iloc[i, j]['location'][0]
            py = shots_df_freeze.iloc[i, j]['location'][1]
            # compute area of the triangle (with indices of points arranged counter clock-wise)
            Area = 0.5 *(-p1y*p2x + p0y*(-p1x + p2x) + p0x*(p1y - p2y) + p1x*p2y)
            # Compute barycentric coordinates s and t of point p by using the analytical
            # solution of the equation system 
            s = 1/(2*Area)*(p0y*p2x - p0x*p2y + (p2y - p0y)*px + (p0x - p2x)*py)
            t = 1/(2*Area)*(p0x*p1y - p0y*p1x + (p0y - p1y)*px + (p1x - p0x)*py)
            # Define condition d to check if P is within the area of the triangle
            if (s>0 and t>0 and 1-s-t>0) :
                # If the condition holds append the player to the list of 
                # teammates or opponents respectively
                if shots_df_freeze.iloc[i, j]['teammate'] == True :
                    teammates.append(1)
                else :
                    opponents.append(1)
    # Add number of teammates within the shot angle as variable
    shots_df.at[i,'number teammates'] = len(teammates)
    # Add number of opponents within the shot angle as variable
    shots_df.at[i,'number opponents'] = len(opponents)

# Check for observations with na values
shots_df['min distance opponents'].isnull().values.any()
shots_df['distance goalkeeper'].isnull().values.any()

# Omit observations with na values
shots_df = shots_df.dropna(subset=['min distance opponents', 'distance goalkeeper'])

# Omit unneeded variables 
shots_df = shots_df.drop(['location', 'freeze_frame'], axis = 1)

# Extract target variable (labels) y from shots_df        
y = np.array(shots_df.loc[:, ['outcome']]) 

# Extract different feature sets X1, X2, X3 from shots_df
X1 = np.array(shots_df.loc[:, ['distance to goal', 'angle to goal']]) # Feature set 1
X2 = np.array(shots_df.loc[:, ['distance to goal', 'angle to goal', 
                               'min distance opponents', 'distance goalkeeper', 
                               'number teammates', 'number opponents']]) # Feature set 2
X3 = np.array(shots_df.drop(['outcome'], axis = 1)) # Feature set 3

# Normalize feature sets X1, X2, X3 (for the neural network)
scaler = MinMaxScaler()
X1_normalized = scaler.fit_transform(X1) # Feature set 1 (normalized)
X2_normalized = scaler.fit_transform(X2) # Feature set 2 (normalized)
X3_normalized = scaler.fit_transform(X3) # Feature set 3 (normalized)

# Load fully pre-processed shots_df from csv. file
shots_df = pd.read_csv('shots_df_pre-processed.csv')


#### Model Implementation #####################################################

# 1) Baseline model
def baseline(y_train, y_test, output = 'auc'):
    # Compute proportion of goals in the training set
    prediction = y_train.mean()
    # Generate predictions on the test set based on the proportion of goals
    predictions = np.array([prediction for i in range(len(y_test))])
    # Generate labels from test set
    labels = np.array(y_test)
    # Compute AUC
    auc = roc_auc_score(y_true=labels, y_score=predictions)
    # Define output of the function
    if output == 'predictions' :
        # Function outputs predictions
        return predictions
    else :
        # Function outputs AUC measure (Default)
        return auc 

# 2) Logistic Regression
logisticregression = LogisticRegression(penalty = 'l2')

# 3) Neural Network
def neuralnetwork(n_layers, n_nodes):
    # Create model
    model = Sequential()
    # Add layers
    for i in range(1, n_layers+1) :
        if i==1 :
            # First layer
            model.add(Dense(n_nodes, input_dim = X.shape[1], activation = 'relu'))
        else :
            # Additional hidden layers
            model.add(Dense(n_nodes, activation = 'relu')) 
    # Output layer
    model.add(Dense(1, activation = 'sigmoid'))
    # Compile model
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
    return model


#### Model Evaluation #########################################################

# 5-fold cross validation (for baseline model)
def cross_val(X, y) :
    # Initialize list to store results
    results = [] 
    # Initialize cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=1) 
    # Perform cross validation (Train-Test split)
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Compute AUC for fold
        auc = baseline(y_train, y_test)
        # Store results
        results.append(auc)
    # Compute mean AUC of cross validation
    mean_auc = np.mean(np.array(results))
    # Return mean AUC
    return mean_auc

# Nested cross validation (for logistic regression and neural network)    
def nested_cross_val(X, y, model) :
    # Initialize list to store outer results
    outer_results = [] 
    # Initialize list to store inner results (best score)
    inner_results_best_score = [] 
    # Initialize list to store inner results (best parameters)
    inner_results_best_params = []
    # Initialize outer cross validation
    cv_outer = KFold(n_splits=5, shuffle=True, random_state=1) 
    # Perform outer cross validation (Train-Test split)
    for train_index, test_index in cv_outer.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Initialize inner cross validation
        cv_inner = KFold(n_splits=5, shuffle=True, random_state=1)
        # Select model
        if model == 'logistic regression' :
            # Initialize model from function implemented above
            model = logisticregression
            # Initialize parameters for grid search 
            param_grid = dict(solver = ['lbfgs', 'liblinear', 'sag', 'newton-cg'], 
                              C = [1e3, 5e3, 1e4, 5e4])
        elif model == 'neural network' :
            # Initialize model from function implemented above
            model = KerasClassifier(build_fn = neuralnetwork, 
                                    epochs = 10, batch_size = 16, verbose = 3) 
            # Initialize parameters for grid search 
            param_grid = dict(n_layers = [1, 2],
                              n_nodes = [6, 12, 18, 24, 30, 36])
        # Perform grid search with inner cross validation (Train-Validation split)
        grid = GridSearchCV(estimator = model, param_grid = param_grid,
                            scoring = 'roc_auc', n_jobs = 1, cv = cv_inner, refit=True)
        grid_result = grid.fit(X = X_train, y = y_train)
        # Store inner results (best score)
        inner_results_best_score.append(grid_result.best_score_) 
        # Store inner results (best parameters)
        inner_results_best_params.append(grid_result.best_params_) 
        # Extract best model from grid search
        best_model = grid_result.best_estimator_ 
        # Make predictions on test set from outer cross validation
        predictions = best_model.predict_proba(X_test)[:, 1] 
        # Generate labels from test set
        labels = np.array(y_test)
        # Compute AUC for fold
        auc = roc_auc_score(y_true=labels, y_score=predictions)
        # Store outer results
        outer_results.append(auc) 
    # Compute mean AUC of outer cross validation
    mean_auc = np.mean(np.array(outer_results)) 
    # Return mean AUC, inner results and outer results 
    return mean_auc, outer_results, inner_results_best_score, inner_results_best_params 


#### Experiment ###############################################################

# Generate input data X from feature sets depending on the experiment conducted 
X = X1 # Feature Set 1
X = X2 # Feature Set 2
X = X3 # Feature Set 3
X = X1_normalized # Feature Set 1 (normalized)
X = X2_normalized # Feature Set 2 (normalized)
X = X3_normalized # Feature Set 3 (normalized)
# Feature sets need to be defined depending on the experiment

# Perform experiments with different models     
cross_val(X, y, 'baseline') # Mean AUC of baseline 
nested_cross_val(X, y, model = 'logistic regression') # Mean AUC of logistic regression
nested_cross_val(X, y, model = 'neural network') # Mean AUC of neural network


