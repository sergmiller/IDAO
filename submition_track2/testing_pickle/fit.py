from sklearn import svm
from sklearn import datasets

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf = svm.SVC()
clf.fit(X, y)

##########################
# SAVE-LOAD using pickle #
##########################
import pickle

# save
with open('model.pkl','wb') as f:
    pickle.dump(clf,f)