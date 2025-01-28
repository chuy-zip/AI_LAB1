import pandas as pd

df = pd.read_csv("Task 2_2/dataset_phishing.csv")

df_legit = df[df["status"] == "legitimate"]
df_phish = df[df["status"] == "phishing"]

print(df_legit)

df_legit80 = df_legit.sample(frac=0.8)
df_phish80 = df_phish.sample(frac=0.8)

# Obtener el 20% restante de los datos legítimos (conjunto de prueba)
df_legit_test = df_legit.drop(df_legit80.index)

# Obtener el 20% restante de los datos de phishing (conjunto de prueba)
df_phish_test = df_phish.drop(df_phish80.index)

# Combinar los conjuntos de prueba
df_test = pd.concat([df_legit_test, df_phish_test])

# Combinar los conjuntos de entrenamiento
df_train = pd.concat([df_legit80, df_phish80])

print(df_train["status"].value_counts(), end= "\n\n")
print(df_test["status"].value_counts(),  end= "\n\n")

df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)