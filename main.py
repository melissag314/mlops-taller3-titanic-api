from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from pandas import DataFrame
from joblib import load

# Cargamos el pipeline entrenado (preprocesamiento + modelo)
model = load("pipeline_titanic.joblib")

app = FastAPI()

# Cada lista interna es un pasajero, con los valores en este orden:
# pclass, age, sibsp, parch, fare, sex, embarked
class DataPredict(BaseModel):
    data_to_predict: list[list] = [[1, 30, 0, 0, 100, 'female', 'C'], [3, None, 1, 2, 15, 'male', 'S']]

@app.post("/predict")
def predict(request: DataPredict):
    try:
        list_data = request.data_to_predict
        df_data = DataFrame(list_data, columns=['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'embarked'])

        prediction = model.predict(df_data)
        return {"prediction": prediction.tolist()}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))