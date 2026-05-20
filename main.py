from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import transcribe
from inference.inference import load_model

app = FastAPI()

@app.on_event("startup")
def startup():
    load_model()

app.include_router(transcribe.router)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def root():
    return FileResponse("frontend/index.html")