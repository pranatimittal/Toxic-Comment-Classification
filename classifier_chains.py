# using classifier chain
from skmultilearn.problem_transform import ClassifierChain
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from data_preprocess import preprocess


features_train, features_test, labels_train, labels_test = preprocess()

parameters = [
    {
        'classifier': [MultinomialNB()],
        'classifier__alpha': [0.7, 1.0]
    },
    {
        'classifier': [SVC()],
        'classifier__kernel': ['rbf'],
        'classifier__C': [1,1000,1000000],
        'classifier__gamma': ['auto']
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [7, 50,100]
    },
]

grid_clf = GridSearchCV(ClassifierChain(), parameters, scoring='accuracy',cv=3)
grid_clf = grid_clf.fit(features_train, labels_train)

print ("Best parameters:\n",grid_clf.best_params_)

# accuracy
print("Accuracy = ",grid_clf.best_score_)
