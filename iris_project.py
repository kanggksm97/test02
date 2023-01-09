from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

Iris = load_iris()

Iris_Data = pd.DataFrame(data= np.c_[Iris['data'], Iris['target']], columns= Iris['feature_names'] + ['target'])
Iris_Data['target'] = Iris_Data['target'].map({0: "setosa", 1:"versicolor", 2:"virginica"})

X_Data = Iris_Data.iloc[:, :-1]
Y_Data = Iris_Data.iloc[:, [-1]]