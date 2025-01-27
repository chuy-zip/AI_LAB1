# Este código fue realizado con el apoyo de la siguiente explicación
# https://towardsdatascience.com/logistic-regression-from-scratch-in-python-ec66603592e2
# La implementación en la guias es un tanto mas compleja e incluye mas calculos (como el loss)
# por lo que esta version esta simplificada pero siempre se cumplen con los conceptos escenciales
# para una regresion logistica

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def normalize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)

df_phish = pd.read_csv("dataset_phishing.csv")

print(df_phish["status"].value_counts(), end="\n\n")
                       
print(df_phish.shape)
print(df_phish.columns)
#La columna status realmente debería de ser 0 o 1 para identyificar si es legitima o phisihing respectivamente

print(df_phish['status'])
# Convertir valores de 'status' a valores binarios
df_phish['status'] = df_phish['status'].map({'legitimate': 0, 'phishing': 1})

print(df_phish['status'].unique())  #[0,1]
print("New phishhhhhhhhhhhhhhhhhh")
print(df_phish["status"])

df = df_phish[["length_url","length_hostname"]]
print("Analyzing features: \n", df)

print(df.shape)


X = df.values
X = normalize(X)
Y = df_phish['status'].values
m = df.shape[0] # la cantidad de registros en el dataframe de padnas
W = np.zeros(df.shape[1])
b = 0 # sesgo 0 de momento

# Hiperparámetros
learning_rate = 0.2  
num_iterations = 1000

for iteration in range(num_iterations):

    # calculo de  z = X * W + b
    z = np.dot(X, W) + b

    # hipotesis/ predicciones
    y_hat = sigmoid(z)

    # gradientes con la foprmula ya derivada
    dw = (1/m) * np.dot(X.T, (y_hat - Y))  # Gradiente W
    db = (1/m) * np.sum(y_hat - Y)  # Gradiente b

    W -= learning_rate * dw
    b -= learning_rate * db

print("Pesos finales:", W)
print("Sesgo final:", b)

# Convertir probabilidades a clases binarias
predictions = (y_hat >= 0.5).astype(int)

# Calcular accuracy
accuracy = np.mean(predictions == Y)
print(f"Precisión del modelo: {accuracy:.2f}")

# Graficar los resultados
plt.scatter(X[:, 0], X[:, 1], c=Y)
plt.xlabel('length_url')
plt.ylabel('length_hostname')
plt.title('Clasificación de phishing')

# Calcular la línea de decisión (frontera)
x_values = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
y_values = -(W[0]*x_values + b)/W[1]
plt.plot(x_values, y_values, 'r', label='Frontera de decisión')
plt.legend()

plt.show()
