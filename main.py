from sklearn import datasets  # will use ideal dataset that we have in sklearn

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline  # to import pipeline
from sklearn.decomposition import PCA  # To perform dimensionality reduction (additional step in pipeline)
from sklearn.preprocessing import StandardScaler  # for scaling step in pipeline
from sklearn.tree import DecisionTreeClassifier  # for ML model here DT model

from sklearn.metrics import accuracy_score  # to check the score

# load the dataset
iris = datasets.load_iris()
x = iris.data
y = iris.target
print(x)
print(y)

# split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# create pipeline flow
# --> first thing we are going to perform is the dimensionality reduction (PCA) pca->principal component analysis and will reduce it to and component to two.
# --> Next thing is to perform the standard scaling (StandardScaler)
# --> third step , want DT classifier (DecisionTreeClassifier)
# --> finally to get some details (Verbose)
pipe = Pipeline([('pca', PCA(n_components=2)), ('std', StandardScaler()), ('dt', DecisionTreeClassifier())], verbose=True)

# to fit the data in the pipe
print(pipe.fit(x_train, y_train))

# check the scoring of the data
print(accuracy_score(y_test, pipe.predict(x_test)))
