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

# Cargar los datos de entrenamiento
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Convertir valores de 'status' a valores binarios
df_train['status'] = df_train['status'].map({'legitimate': 0, 'phishing': 1})
df_test['status'] = df_test['status'].map({'legitimate': 0, 'phishing': 1})

# elegir caracteristicas del data set de training
X_train = df_train[["length_url", "length_hostname"]].values
Y_train = df_train['status'].values

# elegir caracteristicas del data set de prueba
X_test = df_test[["length_url", "length_hostname"]].values
Y_test = df_test['status'].values

X_train = normalize(X_train)
X_test = normalize(X_test)

# paranms
m = X_train.shape[0]  # cantidad de registraos para entrenar
W = np.zeros(X_train.shape[1])  # pesos
b = 0  # sesgo inicial

# Hiperparámetros
learning_rate = 0.2  
num_iterations = 1000

# Entrenamiento del modelo
for iteration in range(num_iterations):

    # calcular z = X * W + b
    z = np.dot(X_train, W) + b

    # Hipótesis / predicciones
    y_hat = sigmoid(z)

    # Gradientes con las fórmulas ya derivadas
    dw = (1 / m) * np.dot(X_train.T, (y_hat - Y_train))  # Gradiente  W
    db = (1 / m) * np.sum(y_hat - Y_train)  # Gradniente b

    # Actualización de parámetros
    W -= learning_rate * dw
    b -= learning_rate * db

print("Pesos finales:", W)
print("Sesgo final:", b)

# prediccion para testing
z_test = np.dot(X_test, W) + b
y_test_hat = sigmoid(z_test)
predictions_test = (y_test_hat >= 0.5).astype(int)

# Métrica de precisión
accuracy_test = np.mean(predictions_test == Y_test)
print(f"Precisión del modelo en el conjunto de prueba: {accuracy_test:.2f}")

# graficar
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test)
plt.xlabel('length_url')
plt.ylabel('length_hostname')
plt.title('Clasificación de phishing (conjunto de prueba propio)')

# linea de frontera
x_values = np.array([np.min(X_test[:, 0]), np.max(X_test[:, 0])])
y_values = -(W[0] * x_values + b) / W[1]
plt.plot(x_values, y_values, 'r', label='Frontera de decisión')
plt.legend()

plt.show()