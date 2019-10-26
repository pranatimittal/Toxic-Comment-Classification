# using ML-KNN
from skmultilearn.adapt import MLkNN
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.model_selection import GridSearchCV
from data_preprocess import preprocess


features_train, features_test, labels_train, labels_test = preprocess()

features_train = lil_matrix(features_train).toarray()
labels_train = lil_matrix(labels_train).toarray()
features_test = lil_matrix(features_test).toarray()

parameters = {'k': [3, 5, 7]}

grid_clf = GridSearchCV(MLkNN(), parameters, scoring='accuracy',cv=3)
grid_clf.fit(features_train, labels_train)

print ("Best parameters:\n",grid_clf.best_params_)

# accuracy
print("Accuracy = ",grid_clf.best_score_)
