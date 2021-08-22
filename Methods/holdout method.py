import pandas
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import svm
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
testSize = 0.33
Xtr, Xtest, Ytr, Ytest = model_selection.train_test_split(X, Y, test_size= testSize, random_state= 21)
clf = svm.SVC(kernel='linear', C=1).fit(Xtr, Ytr)
print(clf.score(Xtest,Ytest))