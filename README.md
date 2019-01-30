# MLProject
This project uses Binary Classification to predict the class of tennis match duration.

The python scripts use scikit-learn library to apply the machine learning algorithms.
https://scikit-learn.org/stable/index.html

I also uploaded the Tennis dataset here for two reasons:
the first one is that I corrected wrong entries of the orginal files uploaded at the following link:
https://github.com/JeffSackmann/tennis_atp
and I only used a part of the historical data and applied the code to the files of the years from 2009-2015

The python scripts are:

1-Features_Extraction.py:
this script cleanse the data and apply map functions on the features to produce the dimensions used for prediction.

in any order you like:
2-LogisticRegression.py:
this script applies Logistic Regression Classifier to predict the target variable

3-RandomForests.py:
this script applies Random Forests Classifier to predict the target variable

4-SVC.py:
this script applies Support Vector Classifier to predict the target variable

5-Voting.py:
this script applies Voting Classifier to predict the target variable
