# #Implementación del algoritmo K-Nearest Neighbors (KNN) para clasificación de URLs legítimas y de phishing
# En este código se hace uso de sklearn, se verificará si este código tiene más accuracy que el código implementado desde cero.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

# Crear el modelo KNN con k=3
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# Entrenar el modelo
knn.fit(X_train, Y_train)

# Realizar predicciones
predictions = knn.predict(X_test)

# Calcular la matriz de confusión manualmente
conf_matrix = confusion_matrix(Y_test, predictions)

# Extraer valores TP, TN, FP, FN de la matriz de confusión
TP = conf_matrix[1, 1]  # Verdaderos positivos
TN = conf_matrix[0, 0]  # Verdaderos negativos
FP = conf_matrix[0, 1]  # Falsos positivos
FN = conf_matrix[1, 0]  # Falsos negativos

# Calcular el accuracy manualmente usando la fórmula proporcionada
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Imprimir resultados
print("Matriz de confusión:")
print(conf_matrix)
print(f"Precisión del modelo (manual): {accuracy:.2f}")

# Visualización de los resultados del conjunto de prueba con la corrección de la leyenda
plt.figure(figsize=(10, 6))

# Graficar cada clase por separado
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

# Visualización de la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Legítimos', 'Phishing'])
disp.plot(cmap='coolwarm')
plt.title('Matriz de Confusión')
plt.show()
