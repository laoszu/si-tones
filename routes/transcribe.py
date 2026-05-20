from fastapi import APIRouter, UploadFile, File
from inference.inference import run_inference

router = APIRouter(prefix="/api")

@router.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    result = run_inference(audio_bytes)
    
    return {"transcription": result}