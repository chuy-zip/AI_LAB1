import pandas as pd
import numpy as np

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

df_phish = pd.read_csv("dataset_phishing.csv")
                       
print(df_phish.shape)
print(df_phish.columns)
#La columna status realmente debería de ser 0 o 1 para identyificar si es legitima o phisihing respectivamente
print(df_phish["status"])

# Convertir valores de 'status' a valores binarios
df_phish['status'] = df_phish['status'].map({'legitimate': 0, 'phishing': 1})
print(df_phish['status'].unique())  # Deberías ver [0, 1]

#remover las columnas no numericas para probar (tal vez reduzca las columasn luego)
df = df_phish.drop(columns=["url", "status"])

#df = df_phish[["length_url","https_token", "punycode"]]

print(df.shape)

# Ahora que tengo 87 columnas necesito 87 pesos w
X = df.values
Y = df_phish['status'].values
W = np.zeros(df.shape[1])
b = 0 # sesgo 0 de momento

# Hiperparámetros
learning_rate = 0.01  
num_iterations = 1000

z = np.dot(X, W) + b

print(z)

# hipotesis/ predicciones
y_hat = sigmoid(z)
print(y_hat)