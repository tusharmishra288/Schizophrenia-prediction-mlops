
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from schizophrenia_prediction.constants import APP_HOST, APP_PORT
from schizophrenia_prediction.pipeline.prediction_pipeline import SchizophreniaData, SchizophreniaClassifier
from schizophrenia_prediction.pipeline.training_pipeline import TrainPipeline
from schizophrenia_prediction.logger import logging


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Disease_Duration: int
        self.Hospitalizations: int
        self.Family_History: int
        self.Substance_Use: int
        self.Suicide_Attempt: int
        self.Positive_Symptom_Score: int
        self.Negative_Symptom_Score: int
        self.GAF_Score: int
        self.Medication_Adherence: int
        

    async def get_schizophrenia_data(self):
        form = await self.request.form()
        self.Disease_Duration = form.get("Disease_Duration")
        self.Hospitalizations = form.get("Hospitalizations")
        self.Family_History = form.get("Family_History")
        self.Substance_Use = form.get("Substance_Use")
        self.Suicide_Attempt = form.get("Suicide_Attempt")
        self.Positive_Symptom_Score = form.get("Positive_Symptom_Score")
        self.Negative_Symptom_Score =  form.get("Negative_Symptom_Score")
        self.GAF_Score = form.get("GAF_Score")
        self.Medication_Adherence = form.get("Medication_Adherence")

@app.get("/", tags=["authentication"],response_class=HTMLResponse)
async def index(request: Request):

    return templates.TemplateResponse(
            "schizophrenia.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def predictRouteClient(request: Request):
    try:
        form_data = DataForm(request)
        await form_data.get_schizophrenia_data()
        logging.info("In prediction pipeline")
        schizophrenia_data = SchizophreniaData(
                                Disease_Duration = form_data.Disease_Duration,
                                Hospitalizations = form_data.Hospitalizations,
                                Family_History = form_data.Family_History,
                                Substance_Use = form_data.Substance_Use,
                                Suicide_Attempt= form_data.Suicide_Attempt,
                                Positive_Symptom_Score= form_data.Positive_Symptom_Score,
                                Negative_Symptom_Score = form_data.Negative_Symptom_Score,
                                GAF_Score= form_data.GAF_Score,
                                Medication_Adherence= form_data.Medication_Adherence
                                )
                
        schizophrenia_df = schizophrenia_data.get_schizophrenia_input_data_frame()

        model_predictor = SchizophreniaClassifier()

        value = model_predictor.predict(dataframe=schizophrenia_df)[0]

        status = None
        if value == 1:
            status = "Schizophreniac"
        else:
            status = "Not Schizophreniac"

        return templates.TemplateResponse(
            "schizophrenia.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)     