import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos de entrenamiento y prueba
df_train = pd.read_csv("Task 2_1/train.csv")
df_test = pd.read_csv("Task 2_1/test.csv")

# Convertir 'status' a valores binarios (0: legitimate, 1: phishing)
df_train['status'] = df_train['status'].map({'legitimate': 0, 'phishing': 1})
df_test['status'] = df_test['status'].map({'legitimate': 0, 'phishing': 1})

# Seleccionar las características y las etiquetas
X_train = df_train[['length_url', 'length_hostname']].values
y_train = df_train['status'].values

X_test = df_test[['length_url', 'length_hostname']].values
y_test = df_test['status'].values

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Legítimo', 'Phishing'], yticklabels=['Legítimo', 'Phishing'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de confusión')
plt.show()

# Informe de clasificación
print("Informe de clasificación:\n", classification_report(y_test, y_pred))

# Visualización de la frontera de decisión
# Visualización de la frontera de decisión con etiquetas claras
plt.figure(figsize=(10, 6))

# Puntos legítimos
legit_points = (y_test == 0)  # Etiqueta 0: legítimo
plt.scatter(X_test[legit_points, 0], X_test[legit_points, 1], 
            c='yellow', label="Legítimos (Reales)", alpha=0.7)

# Puntos phishing
phish_points = (y_test == 1)  # Etiqueta 1: phishing
plt.scatter(X_test[phish_points, 0], X_test[phish_points, 1], 
            c='purple', label="Phishing (Reales)", alpha=0.7)

# Etiquetas y visualización
plt.xlabel('length_url')
plt.ylabel('length_hostname')
plt.title('Clasificación de phishing (Regresión logística)')

# Calcular la frontera de decisión
x_values = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
y_values = -(model.coef_[0, 0] * x_values + model.intercept_[0]) / model.coef_[0, 1]
plt.plot(x_values, y_values, 'r', label='Frontera de decisión')

plt.legend()
plt.show()
