import pickle
import numpy as np
X = np.array([[5.1, 3.5, 1.4, 0.2]])
# load
with open('model.pkl', 'rb') as f:
    clf2 = pickle.load(f)

print(clf2.predict(X[0:1]))