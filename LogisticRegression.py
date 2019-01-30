import pandas as pd
from sklearn.model_selection import train_test_split
from yellowbrick.classifier import ClassificationReport
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#1. Read the dataset
data = pd.read_csv('dataset.csv',header=0 )

#2. Extract the training data and the target variable 
cols = [col for col in data.columns if col not in ['label']]
tdata = data[cols]
target = data['label']
X_train, X_test, y_train, y_test = train_test_split( tdata, target, test_size = 0.15, shuffle=False )
target_names = ['Label -1', 'Label 1']

#3. Apply Logitsic Regression classifier 
lr_clf = LogisticRegression( C = 2.5 , solver = 'newton-cg'  ) 
lr_clf.fit(X_train, y_train)

#4. Predict the response for test dataset
y_pred = lr_clf.predict(X_test)

#5. Print model Accuracy: how often is the classifier correct
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#6. Print the main classification metrics
print(metrics.classification_report(y_test , y_pred, target_names= target_names))

#7. Save the results in a dataframe and csv file
dd = pd.DataFrame(y_pred,y_test)
dd.to_csv("lr_pred.csv")