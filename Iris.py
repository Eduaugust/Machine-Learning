from numpy.lib.polynomial import polyfit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('iris.data.csv')
nomes = [ "Comprimento-sépala", 'Tamanho-sépala', 'Comprimento-pétala', 'Tamanho-pétala', 'Classe',]
data.columns = nomes

"""
print(data.groupby("Classe").size()) # Observando quantas classes e suas quantidades existem
print(data.describe()) # Observando os dados, médias, min, max...
print(pd.DataFrame.corr(data)) # Observando alguma correlação entre os dados
"""
# Treinando
x = data.values[:,0:4]
y = data.values[:,4]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.33, random_state=21, shuffle= True)
modelo = LinearSVC()
modelo.fit(Xtrain, Ytrain)

#Testando
tentativas = modelo.predict(Xtest)
pontuacao = accuracy_score(Ytest, tentativas)
print('Acertou {}%'.format(pontuacao*100))