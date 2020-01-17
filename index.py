import warnings
warnings.simplefilter('ignore')

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib
import numpy as np
import scipy as sp
from IPython.display import display
import mglearn
from sklearn.neighbors import KNeighborsClassifier
# jupyter Notebookで必要
# % matplotlib inline
import matplotlib.pyplot as plt
from scipy import sparse
import sys
import IPython
import sklearn


# 1-1
x = np.array([[1,2,3], [4,5,6]])
# print("x:\n{}".format(x))

# 1-2
eye = np.eye(4)
# print("NumPy array:\n{}".format(eye))

# 1-3
sparse_matrix = sparse.csr_matrix(eye)
# print("\nSciPy sparse CSR matrix:\n{}".format(sparse_matrix))

# 1-4
data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
# print("COO representation:\n{}".format(eye_coo))

# 1-5
x = np.linspace(-10, 10, 100)
y = np.sin(x)
# plt.plot(x, y, marker="x")

# 1-6
data = {'Name': ["John", "Anna", "Peter", "Linda"],
        'Location': ["New York", "Paris", "Berlin", "London"],
        'Age': [24, 13, 53, 33]
        }
data_pandas = pd.DataFrame(data)
# display(data_pandas)

# 1-7
# display(data_pandas[data_pandas.Age > 30])

# 1-8
# print("Python version: {}".format(sys.version))
# print("pandas version: {}".format(pd.__version__))
# print("matplotlib version: {}".format(matplotlib.__version__))
# print("NumPy version: {}".format(np.__version__))
# print("SciPy version: {}".format(sp.__version__))
# print("IPython version: {}".format(IPython.__version__))
# print("scikit-learn version: {}".format(sklearn.__version__))

# 1-9
iris_dataset = load_iris()

# 1-10
# print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# 1-11
# print(iris_dataset['DESCR'][:193] + "\n...")

# 1-12
# print("Target names: {}".format(iris_dataset['target_names']))

# 1-13
# print("Feature name: \n{}".format(iris_dataset['feature_names']))

# 1-14
# print("Type of data: {}".format(type(iris_dataset['data'])))

# 1-15
# print("Shape of data: {}".format(iris_dataset['data'].shape))

# 1-16
# print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))

# 1-17
# print("Type of target: {}".format(type(iris_dataset['target'])))

# 1-18
# print("Shape of target: {}".format(iris_dataset['target'].shape))

# 1-19
# print("Target:\n{}".format(iris_dataset['target']))

# 1-20
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# 1-21
# print("X_train shape: {}".format(X_train.shape))
# print("y_train shape: {}".format(y_train.shape))

# 1-22
# print("X_test shape: {}".format(X_test.shape))
# print("y_test shape: {}".format(y_test.shape))

# 1-23
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                        hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

# 1-24
knn = KNeighborsClassifier(n_neighbors=1)

# 1-25
knn.fit(X_train, y_train)
# print(knn)

# 1-26
X_new = np.array([[5, 2.9, 1, 0.2]])
# print("X_new.shape: {}".format(X_new.shape))

# 1-27
prediction = knn.predict(X_new)
# print("Prediction: {}".format(prediction))
# print("Predicted target name: {}".format(
#     iris_dataset['target_names'][prediction]))

# 1-28
y_pred = knn.predict(X_test)
# print("Test set predictions:\n {}".format(y_pred))


# 1-29
# print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

# 1-30
# print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# 1-31
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)
knn= KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

# 2-1
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legent(["Class 0", "Class 1"], loc=4)
plt.legent("First feature")
plt.legent("Second feature")
print("X.shape: {}".format(X.shape))


