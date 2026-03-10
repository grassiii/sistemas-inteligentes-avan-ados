'''
Este exemplo demonstra o procedimento básico para normalizar
um novo valor, obedecendo ao modelo normalizador dos dados
originais
'''

import pickle
import numpy as np

novo_dado = [[2200]]

# Abrir o modelo normalizador
scaler_model = pickle.load(open('scaler_model.pkl', 'rb'))

# Normalizar o novo dado obedecendo ao modelo prévio
novo_dado_norm = scaler_model.transform(novo_dado)
print(novo_dado_norm)

# Atividade
# 1. Receber um valor normalizado
val_norm = [[0.37]]

# Reverter de normalizado para natural usando o modelo 
# scaler_model
val_desnorm = scaler_model.inverse_transform(val_norm)
print(val_desnorm)

