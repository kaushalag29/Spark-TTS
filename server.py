import os
import sys
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import uvicorn
import numpy as np

# Add Spark-TTS to path
sys.path.append(os.path.dirname(__file__))

from cli.SparkTTS import SparkTTS

app = FastAPI(title="Spark-TTS API Server")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Detect device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
logger.info(f"Using device: {device}")

# --- Pydantic Models for Requests ---
class TTSRequest(BaseModel):
    text: str
    prompt_speech_path: str
    output_path: str
    temperature: Optional[float] = None
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.95

# --- Model Loading ---
spark_model = None
try:
    logger.info("--- Starting Spark-TTS Model Loading ---")
    model_dir = os.path.join(os.path.dirname(__file__), 'pretrained_models/Spark-TTS-0.5B')
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Spark-TTS model directory not found at {model_dir}")
    
    spark_model = SparkTTS(model_dir=Path(model_dir), device=torch.device(device))
    logger.info("--- Spark-TTS Model Loaded Successfully ---")
except Exception as e:
    logger.error(f"--- FATAL: Error during Spark-TTS model loading ---", exc_info=True)
    spark_model = None # Ensure model is None if loading fails

@app.post("/generate-audio")
async def generate_audio(request: TTSRequest):
    """Generate audio using Spark-TTS, trying different temperatures if needed."""
    if spark_model is None:
        raise HTTPException(status_code=500, detail="Spark-TTS model not loaded.")

    logger.info(f"Generating Spark-TTS audio for text: '{request.text}'")
    logger.info(f"Using prompt audio: {request.prompt_speech_path}")
    
    if not os.path.exists(request.prompt_speech_path):
        raise HTTPException(status_code=404, detail=f"Prompt audio file not found: {request.prompt_speech_path}")

    wav_numpy = None
    temperatures_to_try = [request.temperature] if request.temperature is not None else [0.8, 0.7, 0.6, 0.5]

    for temp in temperatures_to_try:
        try:
            logger.info(f"Trying TTS generation with temperature: {temp}")
            with torch.no_grad():
                # This method returns a numpy array, as confirmed by the error logs.
                wav_output = spark_model.inference(
                    text=request.text,
                    prompt_speech_path=Path(request.prompt_speech_path),
                    temperature=temp,
                    top_k=request.top_k,
                    top_p=request.top_p
                )
            
            if isinstance(wav_output, np.ndarray) and wav_output.size > 0:
                wav_numpy = wav_output
                logger.info(f"TTS generation successful with temperature: {temp}")
                break
            else:
                logger.warning(f"TTS generation with temperature {temp} produced empty or invalid output.")
                wav_numpy = None

        except Exception as e:
            logger.error(f"Error during inference at temperature {temp}: {e}", exc_info=True)
            wav_numpy = None
    
    if wav_numpy is None:
        logger.error("TTS generation failed for all attempted temperatures.")
        raise HTTPException(status_code=500, detail="Spark-TTS generation failed for all temperatures.")

    try:
        # Convert the numpy array to a torch tensor for saving.
        # torchaudio.save requires a tensor of shape [channels, samples].
        wav_tensor = torch.from_numpy(wav_numpy).unsqueeze(0)
        torchaudio.save(request.output_path, wav_tensor, spark_model.sample_rate)

        if not os.path.exists(request.output_path) or os.path.getsize(request.output_path) == 0:
            raise HTTPException(status_code=500, detail="Spark-TTS failed to save the audio file (file is empty).")
            
        return {"status": "success", "message": f"Spark-TTS audio saved to {request.output_path}"}
    except Exception as e:
        logger.error(f"Error saving audio file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if spark_model is not None else "unhealthy",
        "model": "spark-tts",
        "device": device,
        "model_loaded": spark_model is not None,
    }

@app.post("/shutdown")
async def shutdown():
    logger.info("Shutdown request received for Spark-TTS server")
    os._exit(0)
    return {"status": "shutdown", "message": "Server shutting down"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010) 