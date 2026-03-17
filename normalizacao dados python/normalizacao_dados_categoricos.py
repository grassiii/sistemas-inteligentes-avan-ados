# Normalização de dados categórico
# Dados categóriso ordinais: LabelEncoding

'''
import numpy as np
from sklearn.preprocessing import LabelEncoder

cores = ['Vermelho', 'Azul', 'Verde', 'Azul']
# Instanciar o codificador
encoder = LabelEncoder()

cores_codificadas = encoder.fit_transform(cores)
#print(cores)
#print(cores_codificadas)

#print(encoder.classes_)
#print(encoder.classes_[2])

'''

# Dados categóricos Nominais: One hot Encoding
import pandas as pd

# Criar um dataframe com os dados a serem normalizados
df = pd.DataFrame(
        {
            'cor':['Vermelho', 'Azul', 'Verde', 'Azul']
        }
        )

print(df['cor'])
cor_normalizada = pd.get_dummies(df, prefix='cor', prefix_sep='_', dtype=int)
print(cor_normalizada) # Mostra os dados convertidos em "dummies"

# Processando dados one hot encoded
# Criar (extrair) uma instancia com o dados codificados
instancia = cor_normalizada.iloc[1]
print('Valor da instância: ', instancia)
# Converter a instancia dm dataframe
df_instancia = instancia.to_frame().T
print(df_instancia)

# Converter de dataframe normalizado para dataframe desnormalizado

df_desnormalizado = pd.from_dummies(df_instancia, sep='_')
print(df_desnormalizado)

