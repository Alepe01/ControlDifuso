import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Cambia esta línea por la ruta completa si es necesario
file_path = "c:/Users/lepea/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/Documents/Programas Phyton/ControlDifuso/sensor_readings_4.data"

# 1. Cargar datos
df = pd.read_csv(file_path, header=None)
df.columns = ['front', 'left', 'right', 'back', 'action']

# 2. EDA
print("Primeras filas:")
print(df.head())

print("\nDistribución de clases:")
print(df['action'].value_counts())

sns.countplot(y='action', data=df)
plt.title("Distribución de clases")
plt.show()

# 3. Preprocesamiento
X = df[['front', 'left', 'right', 'back']]
y = df['action']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 4. División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Entrenamiento
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluación
y_pred = model.predict(X_test)
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()

# 7. Validación cruzada
scores = cross_val_score(model, X, y_encoded, cv=5)
print(f"\nExactitud media (cross-val): {scores.mean():.4f}")
