#import packages
import pydotplus
from sklearn.tree import export_graphviz
from six import StringIO
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

# Read dataset
dataset = pd.read_csv("tree1.csv")

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, 5]

# Label encoder
le = LabelEncoder()
x = x.apply(le.fit_transform)
print(x)

# 1 1 0 0

# import Decesion Tree Classifier

# Create decision tree classifer object
regressor = DecisionTreeClassifier()
# Train model
regressor.fit(x.iloc[:, 1:5], y)

x_in = np.array([1, 1, 0, 0])
y_pred = regressor.predict([x_in])

print(y_pred)

#from IPython.display import Image

dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data, filled=True,
                rounded=True, special_characters=True)

# Draw Graph
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Show graph & Create png File
graph.write_png("tree.png")

# see the tree in .spyder3 folder
