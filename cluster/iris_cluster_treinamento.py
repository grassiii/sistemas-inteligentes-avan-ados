import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

#Abrir o arquivo de dados
dados = pd.read_csv('iris.csv', sep=';')

#Separar atributos numéricos dos atributos categóricos
dados_num = dados.drop(columns=['class'])
dados_cat = dados['class']

#Normalizar dados numéricos
scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)
#Salvar o modelo normalizador
pickle.dump(normalizador, open('normalizador_iris.pkl', 'wb'))
#Normalizar os dados numéricos
dados_num_norm = normalizador.fit_transform(dados_num)

dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep='_', dtype=int)

#Transformar o dados_num_norm em um DataFrame
dados_num_norm = pd.DataFrame(dados_num_norm, columns = dados_num.columns)

#Recompor o DataFrame com todos os dados
dados_norm = dados_num_norm.join(dados_cat_norm, how='left')

print(dados_norm.head(10))