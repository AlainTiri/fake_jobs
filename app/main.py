from app.model.predict import Model_SVM
from fastapi import FastAPI


app = FastAPI()
model = Model_SVM()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/predict/{description}")
def predict(description: str):
    return {"description": description, "p": model.predict_svm(description)}

