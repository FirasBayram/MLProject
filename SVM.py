import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import  GridSearchCV
from sklearn.svm import SVC

#1. Read the dataset
data = pd.read_csv('dataset.csv',header=0 )

#2. Extract the training data and the target variable 
cols = [col for col in data.columns if col not in ['label']]
tdata = data[cols]
target = data['label']
X_train, X_test, y_train, y_test = train_test_split( tdata, target, test_size = 0.15, shuffle=False )
target_names = ['Label -1', 'Label 1']

#3. Use the GridSearch to tune the parameter values for the classifier.
#3.a. define the setof the parameters and their values to be passed to the classifier
grid_param = {  
    'C': [ 2 , 3 , 5 ],
    'kernel': ['rbf' ],
    'gamma' : [ 0.001 , 0.0001 , 0.0005]
}
classifier = SVC()

#3.b. apply GridSearchCV to the classifier with the specified values
gd_sr = GridSearchCV(estimator=classifier,  
    param_grid=grid_param,
    scoring='accuracy',
    cv= 3
    )
gd_sr.fit(X_train, y_train)  

#4. Print the best parameteres and the best score
best_parameters = gd_sr.best_params_  
best_result = gd_sr.best_score_
print(best_parameters)  
print(best_result)

