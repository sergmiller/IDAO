import pickle
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target
# load
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

print(clf2.predict(X[0:1]))