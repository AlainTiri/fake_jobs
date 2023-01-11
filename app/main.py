import app.model.predict as ml
from fastapi import FastAPI
import app.improvement.to_bigquery as to_BigQ


app = FastAPI()
model = ml.Model_SVM()
bigqueryHelper = to_BigQ.ToBigQuery()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/job/{description}")
def read_item(description: str):
    prediction = model.predict(description)
    answer = "Fake job" if prediction[0] == 1 else "True job"
    try:
        print(bigqueryHelper.sendToBQ(description, prediction[0]))
    except Exception:
        print("Error to BigQuery")

    return {"prediction": answer, "description": description}
