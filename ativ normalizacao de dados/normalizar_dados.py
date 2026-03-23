import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Instanciando os scalers
scaler_label = MinMaxScaler()
scaler_onehot = MinMaxScaler()

# Lendo o arquivo dos dados csv com pandas
df_label = pd.read_csv('dados_normalizar.csv', sep=';', decimal=',')

# Separando as colunas numéricas para normalizar
colunas_normalizar_label = ['idade','altura', 'Peso']

# Transformando a coluna string "sexo" em númerica (1 e 0) com Label Encoding
df_label['sexo'] = df_label['sexo'].map({'F': 1, 'M': 0})

# Aplicando o scaler
df_label[colunas_normalizar_label] = scaler_label.fit_transform(df_label[colunas_normalizar_label])

# Normalizando os dados com MinMaxScaler e One Hot Encoding

df_onehot = pd.read_csv('dados_normalizar.csv', sep=';', decimal=',')
# Transformando a coluna string "sexo" em colunas "sexo_F" e "sexo_M", com valores 0 ou 1
df_onehot = pd.get_dummies(df_onehot, columns=['sexo'], dtype=int)

colunas_normalizar_onehot = ['idade', 'altura', 'Peso', 'sexo_F', 'sexo_M']
df_onehot[colunas_normalizar_onehot] = scaler_onehot.fit_transform(df_onehot[colunas_normalizar_onehot])

print('\nResultado dos Dados com Label Encoding\n', df_label)
print('\nResultado dos Dados com One Hot Encoding\n', df_onehot)

# Revertendo os Dados com Label Encoding Normalizados

dados_originais_label = scaler_label.inverse_transform(df_label[colunas_normalizar_label])
df_label_desnormalizado = df_label.copy()
df_label_desnormalizado[colunas_normalizar_label] = dados_originais_label

df_label_desnormalizado['sexo'] = df_label_desnormalizado['sexo'].map({1: 'F', 0: 'M'})

print('\nResultado dos Dados com Label Encoding Desnormalizados\n', df_label_desnormalizado)

dados_originais_onehot = scaler_onehot.inverse_transform(df_onehot[colunas_normalizar_onehot])
df_onehot_desnormalizado = df_onehot.copy()
df_onehot_desnormalizado[colunas_normalizar_onehot] = dados_originais_onehot

colunas_sexo = ['sexo_F', 'sexo_M']
df_onehot_desnormalizado['sexo'] = df_onehot_desnormalizado[colunas_sexo].idxmax(axis=1)

print(df_onehot_desnormalizado['sexo'])

df_onehot_desnormalizado['sexo'] = df_onehot_desnormalizado['sexo'].str.replace('sexo_', '')

df_onehot_desnormalizado = df_onehot_desnormalizado.drop(columns=colunas_sexo)

print('\nResultado dos Dados com One Hot Encoding Desnormalizados\n', df_onehot_desnormalizado)
