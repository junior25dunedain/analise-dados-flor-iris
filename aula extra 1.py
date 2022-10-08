import pandas as pd
import statistics as sts
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('iris.csv', names = ['sepala comprimento','sepala largura','petala comprimento','petala largura','resultado'])

caracteristica = ['sepala comprimento','sepala largura','petala comprimento','petala largura']

########### tratamento dos dados a partir da variancia ####
x = data.loc[:,caracteristica].values
y = data.iloc[:,-1].values

sc = x[:,0]
sl = x[:,1]
pc = x[:,2]
pl = x[:,3]

variancia = np.array([sts.variance(sc),sts.variance(sl),sts.variance(pc),sts.variance(pl)])



fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1)
ax.set_xlabel('sepala comprimento', fontsize= 15)
ax.set_ylabel('petala comprimento', fontsize= 15)
ax.set_title('Representação em duas dimensões',fontsize=20)
resultados = ['Iris-setosa','Iris-versicolor','Iris-virginica']
cores = ['r','g','b']
for saida,cor in zip(resultados,cores):
    indice = data['resultado'] == saida
    ax.scatter(data.loc[indice,'sepala comprimento'],data.loc[indice,'petala comprimento'], c = cor, s = 50)

ax.legend(resultados)
ax.grid()
plt.show()

############## tratamento de dados a partir da correlação #########

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
alvo = label.fit_transform(data['resultado'])
alvo = pd.DataFrame(alvo, columns= ['alvo'])

base = data.iloc[:,:-1]
base = pd.concat([base,alvo],axis=1)

correlacao = base.corr()

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1)
ax.set_xlabel('sepala comprimento', fontsize= 15)
ax.set_ylabel('petala largura', fontsize= 15)
ax.set_title('Representação em duas dimensões',fontsize=20)
resultados = ['Iris-setosa','Iris-versicolor','Iris-virginica']
cores = ['r','g','b']
for saida,cor in zip([0,1,2],cores):
    indice = base['alvo'] == saida
    ax.scatter(base.loc[indice,'sepala comprimento'],base.loc[indice,'petala largura'], c = cor, s = 50)

ax.legend(resultados)
ax.grid()
plt.show()

##### tratamento de dados a partir da analise de componenetes principais (PCA) #####

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

x1 = x
y1 = y

x1 = StandardScaler().fit_transform(x1) # normalizando os atributos de entrada

pca = PCA(n_components=None)
principalcomponents = pca.fit_transform(x1)

principalDf = pd.DataFrame(data= principalcomponents, columns=['componente principal 1','componente principal 2','componente principal 3','componente principal 4'])

finalD = pd.concat([principalDf,data['resultado']], axis=1)

print(pca.components_)

print(pca.explained_variance_ratio_)

# visualizar os dados
pca2 = PCA(n_components=2)
principalcomponents2 = pca2.fit_transform(x1)

principalDf2 = pd.DataFrame(data= principalcomponents2, columns= ['componente principal 1','componente principal 2'])

finalD2 = pd.concat([principalDf2,data['resultado']], axis=1)

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(1,1,1)
ax.set_xlabel('componente principal 1', fontsize= 15)
ax.set_ylabel('componente principal 2', fontsize= 15)
ax.set_title('2 componentes PCA',fontsize=20)
resultados = ['Iris-setosa','Iris-versicolor','Iris-virginica']
cores = ['r','g','b']
for saida,cor in zip(resultados,cores):
    indice = finalD2['resultado'] == saida
    ax.scatter(finalD2.loc[indice,'componente principal 1'],finalD2.loc[indice,'componente principal 2'], c = cor, s = 50)

ax.legend(resultados)
ax.grid()
plt.show()

# visualizar o grafico fasorial
label2 = LabelEncoder()
y2 = label2.fit_transform(y)

def meuplot(score, coeff, labels = None):
    n = coeff.shape[0]
    for i in range(n):
        plt.arrow(0,0,coeff[i,0],coeff[i,1], color = 'r', alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]*1.15, coeff[i,1]+1.15, "var"+str(i+1), color= 'g')
        else:
            plt.text(coeff[i,0] * 1.15, coeff[i, 1] + 1.15, labels[i], color='g',halign="center" )
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel('Componente principal 1 ({:.2f})'.format(pca2.explained_variance_ratio_[0]))
    plt.ylabel('Componente principal 2 ({:.2f})'.format(pca2.explained_variance_ratio_[1]))

    plt.grid()

meuplot(principalcomponents2,np.transpose(pca2.components_))
plt.show()