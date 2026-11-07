# train.py
# Crea, entrena y guarda el pipeline del Titanic (basado en el Taller 1, punto 1).

import seaborn as sns
import joblib
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# 1. Cargar los datos del Titanic
datos = sns.load_dataset("titanic")

columnas_numericas = ["pclass", "age", "sibsp", "parch", "fare"]
columnas_categoricas = ["sex", "embarked"]

X = datos[columnas_numericas + columnas_categoricas]
y = datos["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Preprocesamiento de cada tipo de columna
transformador_numerico = Pipeline(steps=[
    ("imputar", SimpleImputer(strategy="median")),
    ("escalar", StandardScaler())
])

transformador_categorico = Pipeline(steps=[
    ("imputar", SimpleImputer(strategy="most_frequent")),
    ("codificar", OneHotEncoder(handle_unknown="ignore"))
])

preprocesador = ColumnTransformer(transformers=[
    ("numericas", transformador_numerico, columnas_numericas),
    ("categoricas", transformador_categorico, columnas_categoricas)
])

# 3. Modelo con sus hiperparámetros configurados
modelo = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

# 4. Pipeline completo: preprocesamiento + modelo
pipeline = Pipeline(steps=[
    ("preprocesar", preprocesador),
    ("modelo", modelo)
])

# 5. Entrenar y evaluar
pipeline.fit(X_train, y_train)
exactitud = pipeline.score(X_test, y_test)

print(pipeline)
print("Exactitud en prueba:", round(exactitud, 3))

# 6. Guardar el pipeline completo en un archivo
joblib.dump(pipeline, "pipeline_titanic.joblib")
print("Pipeline guardado en pipeline_titanic.joblib")
print("Versión de scikit-learn:", sklearn.__version__)