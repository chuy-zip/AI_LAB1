#Implementación del algoritmo K-Nearest Neighbors (KNN) para clasificación de URLs legítimas y de phishing, desde cero.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Función para calcular la distancia euclidiana
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Implementación del algoritmo KNN
class KNearestNeighbors:
    # se define k como 3 por defecto
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
    
    def predict(self, X_test):
        predictions = []
        for point in X_test:
            distances = [euclidean_distance(point, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.Y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions)

# Función para calcular la matriz de confusión
def compute_confusion_matrix(y_true, y_pred):
    classes = np.unique(y_true)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1
    return matrix

# Función para calcular el accuracy, se extrajo del documento guía del dataset
def compute_accuracy(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return (TP + TN) / (TP + TN + FP + FN)

# Cargar los datos de entrenamiento y prueba
df_train = pd.read_csv("Task 2_2/train.csv")
df_test = pd.read_csv("Task 2_2/test.csv")

# Convertir valores de 'status' a valores binarios
df_train['status'] = df_train['status'].map({'legitimate': 0, 'phishing': 1})
df_test['status'] = df_test['status'].map({'legitimate': 0, 'phishing': 1})

# Seleccionar características y etiquetas
X_train = df_train[["length_url", "length_hostname"]].values
Y_train = df_train['status'].values
X_test = df_test[["length_url", "length_hostname"]].values
Y_test = df_test['status'].values

# Entrenar el modelo KNN
k = 3  # Puedes cambiar el valor de k según sea necesario
knn = KNearestNeighbors(k=k)
knn.fit(X_train, Y_train)

# Realizar predicciones
predictions = knn.predict(X_test)

# Calcular la matriz de confusión y el accuracy
conf_matrix = compute_confusion_matrix(Y_test, predictions)
accuracy = compute_accuracy(Y_test, predictions)

# Imprimir resultados
print("Matriz de confusión:")
print(conf_matrix)
print(f"Precisión del modelo: {accuracy:.2f}")

# Visualización de los resultados del conjunto de prueba
plt.figure(figsize=(10, 6))
# Graficar los datos predichos con colores según la clase
for class_label, color, label_name in zip([0, 1], ['blue', 'red'], ['Legítimos', 'Phishing']):
    plt.scatter(
        X_test[predictions == class_label, 0], 
        X_test[predictions == class_label, 1], 
        color=color, 
        label=label_name, 
        alpha=0.8
    )

plt.xlabel('Longitud del URL')
plt.ylabel('Longitud del Hostname')
plt.title(f'Clasificación de sitios de phishing con KNN (k={k})')
plt.legend()  # Añade la leyenda correctamente configurada
plt.show()

# Gráfica de la matriz de confusión
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(conf_matrix, cmap='coolwarm')
fig.colorbar(cax)
for (i, j), val in np.ndenumerate(conf_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center', color='black')
ax.set_xlabel('Predicción')
ax.set_ylabel('Valor Real')
ax.set_title('Matriz de Confusión')
plt.xticks([0, 1], ['Legítimos', 'Phishing'])
plt.yticks([0, 1], ['Legítimos', 'Phishing'])
plt.show()
