# mlops-taller3-titanic-api

API con FastAPI que predice si un pasajero del Titanic sobrevive, usando un pipeline de scikit-learn (preprocesamiento + modelo). Taller 3 de MLOps, Universidad EIA.

## Archivos

- `train.py`: crea, entrena y guarda el pipeline en `pipeline_titanic.joblib`.
- `pipeline_titanic.joblib`: pipeline entrenado (imputación, escalado, One-Hot y regresión logística).
- `main.py`: API con el endpoint `POST /predict`.
- `requirements.txt`: dependencias con versiones fijas.

## Entrada del endpoint

Lista de pasajeros; cada pasajero es una lista con los valores en este orden:
`pclass, age, sibsp, parch, fare, sex, embarked`

Ejemplo:

```json
{
  "data_to_predict": [[1, 30, 0, 0, 100, "female", "C"], [3, null, 1, 2, 15, "male", "S"]]
}
```

Respuesta: `{"prediction": [1, 0]}` (1 = sobrevive, 0 = no sobrevive).

## Ejecución local

```bash
conda create -n titanic python=3.11
conda activate titanic
pip install -r requirements.txt
python train.py
uvicorn main:app --reload
```

Abrir `http://127.0.0.1:8000/docs`.

## Ejecución en servidor (EC2)

```bash
git clone https://github.com/melissag314/mlops-taller3-titanic-api.git
cd mlops-taller3-titanic-api
conda create -n titanic python=3.11
conda activate titanic
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Abrir `http://IP_PUBLICA:8000/docs`. El puerto 8000 debe estar abierto en el security group.