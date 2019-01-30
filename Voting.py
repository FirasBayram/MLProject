import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#1. Read the dataset
data = pd.read_csv('dataset.csv',header=0 )

#2. Extract the training data and the target variable 
cols = [col for col in data.columns if col not in ['label']]
tdata = data[cols]
target = data['label']
X_train, X_test, y_train, y_test = train_test_split( tdata, target, test_size = 0.15, shuffle=False )
target_names = ['Label -1', 'Label 1']

#3. Apply Logitsic Regression, Random Forests and Voting classifiers
rnd_clf = RandomForestClassifier(n_estimators= 100 , criterion='gini',  max_leaf_nodes=16)
log_clf = LogisticRegression( C = 2.5 , solver = 'newton-cg' )
voting_clf = VotingClassifier(estimators=[ ('lr', log_clf),('rf', rnd_clf) ], voting='soft' )
voting_clf.fit(X_train, y_train)

#4. Predict the response for test dataset for each classifier
for clf in (log_clf , rnd_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, metrics.accuracy_score(y_test, y_pred))
    print(metrics.classification_report(y_test , y_pred))


#5. Save the results in a dataframe and csv file
dd = pd.DataFrame(y_pred,y_test)
dd.to_csv("v_pred.csv")