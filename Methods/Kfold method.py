import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from numpy import mean
from numpy import std
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
kf = model_selection.KFold(n_splits= 10, shuffle=True, random_state=21)
model = LogisticRegression()
result = model_selection.cross_val_score(model, X, Y, cv = kf)
print(mean(result), std(result))