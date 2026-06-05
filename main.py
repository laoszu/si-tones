from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from routes import transcribe
from inference.inference import load_model

app = FastAPI()

@app.on_event("startup")
def startup():
    load_model()

app.include_router(transcribe.router, prefix="/api")

app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

@app.get("/")
def root():
    return FileResponse("frontend/dist/index.html")

@app.get("/{full_path:path}")
def catch_all(full_path: str):
    return FileResponse("frontend/dist/index.html")