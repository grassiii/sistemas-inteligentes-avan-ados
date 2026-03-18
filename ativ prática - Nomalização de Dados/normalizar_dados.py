import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Ainda não finalizado, é necessário adicionar um método para adicionar mais dados

# Instanciando o scaler
scaler = MinMaxScaler()

# Normalizando os dados com MinMaxScaler e Label Encoding

# Lendo o arquivo dos dados csv com pandas
df_label = pd.read_csv('dados_normalizar.csv', sep=';', decimal=',')

# Separando as colunas numéricas para normalizar
colunas_normalizar_label = ['idade','altura', 'Peso']

# Transformando a coluna string "sexo" em númerica (1 e 0) com Label Encoding
df_label['sexo'] = df_label['sexo'].map({'F': 1, 'M': 0})

# Aplicando o scaler
dados_normalizados_label = scaler.fit_transform(df_label[colunas_normalizar_label])

# Normalizando os dados com MinMaxScaler e One Hot Encoding

df_onehot = pd.read_csv('dados_normalizar.csv', sep=';', decimal=',')
# Transformando a coluna string "sexo" em colunas "sexo_F" e "sexo_M", com valores 0 ou 1
df_onehot = pd.get_dummies(df_onehot, columns=['sexo'], dtype=int)

colunas_normalizar_onehot = ['idade', 'altura', 'Peso', 'sexo_F', 'sexo_M']
dados_normalizados_onehot = scaler.fit_transform(df_onehot[colunas_normalizar_onehot])

print('\nResultado dos Dados com Label Encoding\n', df_label.head())
print('\nResultado dos Dados com One Hot Encoding\n', df_onehot.head())