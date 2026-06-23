import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from TextSummarizer.pipeline.prediction import PredictionPipeline

app = FastAPI()

# Load model once at startup
predictor = PredictionPipeline()


class TextRequest(BaseModel):
    text: str = "This is a test document for summarization."

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")


@app.get("/train")
async def train():
    try:
        main_file = os.path.join(BASE_DIR, "main.py")
        os.system(f"python {main_file}")
        return {"message": "Training successful!"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
async def predict_route(request: TextRequest):
    try:
        summary = predictor.predict(request.text)

        return {
            "input_text": request.text,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)