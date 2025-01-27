import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
df_phish = pd.read_csv("dataset_phishing.csv")

# Convertir la columna 'status' a valores binarios (0 para 'legitimate', 1 para 'phishing')
df_phish['status'] = df_phish['status'].map({'legitimate': 0, 'phishing': 1})

# Seleccionar las dos columnas relevantes
df = df_phish[['length_url', 'length_hostname']]
X = df.values  # Características
y = df_phish['status'].values  # Etiquetas

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Estandarizar (normalizar) los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:\n", conf_matrix)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Legítimo', 'Phishing'], yticklabels=['Legítimo', 'Phishing'])
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de confusión')
plt.show()

# Informe de clasificación
print("Informe de clasificación:\n", classification_report(y_test, y_pred))

# Visualización de los datos y la frontera de decisión
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', label="Datos reales")
plt.xlabel('length_url')
plt.ylabel('length_hostname')
plt.title('Clasificación de phishing (Regresión logística)')

# Calcular la frontera de decisión
import numpy as np
x_values = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)
y_values = -(model.coef_[0, 0] * x_values + model.intercept_[0]) / model.coef_[0, 1]
plt.plot(x_values, y_values, 'r', label='Frontera de decisión')

plt.legend()
plt.show()
